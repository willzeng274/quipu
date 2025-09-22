
use std::error::Error;
use std::path::{Path, PathBuf};
use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
use tokio::runtime::Builder;
use jemallocator::Jemalloc;
use jemalloc_ctl::{stats, epoch};

#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use tokio::sync::{mpsc, Mutex};
use uuid::Uuid;
use tracing::info;

const DEFAULT_CHANNEL_CAPACITY: usize = 100;
const MAX_TOKENS_PER_CHUNK: usize = 256;

#[derive(Debug, Clone)]
struct FileInfo {
    id: String,
    path: PathBuf,
    content: String,
}

#[derive(Debug, Clone)]
struct ProcessedFile {
    id: String,
    path: PathBuf,
    content: String,
}

#[derive(Debug, Clone)]
struct ChunkJob {
    file_id: String,
    file_path: PathBuf,
    idx: usize,
    text: String,
}

#[derive(Debug, Clone)]
struct EmbeddedChunk {
    #[allow(dead_code)] // for the vector database integration
    file_id: String,
    file_path: PathBuf,
    idx: usize,
    embedding: Vec<f32>,
}

fn spawn_file_reader_worker(
    handle: tokio::runtime::Handle,
    rx: Arc<Mutex<mpsc::Receiver<PathBuf>>>,
    tx: mpsc::Sender<FileInfo>,
    worker_id: usize,
    content_queue_depth: Arc<AtomicUsize>,
    paths_queue_depth: Arc<AtomicUsize>,
) {
    tokio::task::spawn_blocking(move || {
        loop {
            let maybe_path = handle.block_on(async {
                let mut guard = rx.lock().await;
                guard.recv().await
            });
            let Some(path) = maybe_path else { break };

            println!("Reader {}: processing {}", worker_id, path.display());

            paths_queue_depth.fetch_sub(1, Ordering::Relaxed);

            let file_id = Uuid::new_v4().to_string();
            match fs::read_to_string(&path) {
                Ok(content) => {
                    content_queue_depth.fetch_add(1, Ordering::Relaxed);
                    let _ = handle.block_on(tx.send(FileInfo { id: file_id, path: path.clone(), content }));
                }
                Err(e) => eprintln!("Reader {} error reading file: {}", worker_id, e),
            }
        }
    });
}

fn spawn_preprocessing_worker(
    handle: tokio::runtime::Handle,
    rx: Arc<Mutex<mpsc::Receiver<FileInfo>>>,
    tx: mpsc::Sender<ProcessedFile>,
    worker_id: usize,
    content_queue_depth: Arc<AtomicUsize>,
    processed_queue_depth: Arc<AtomicUsize>,
) {
    tokio::task::spawn_blocking(move || {
        loop {
            let maybe_file = handle.block_on(async {
                let mut guard = rx.lock().await;
                guard.recv().await
            });
            let Some(file_info) = maybe_file else { break };

            println!("Preprocessor {}: processing {}", worker_id, file_info.path.display());

            content_queue_depth.fetch_sub(1, Ordering::Relaxed);
            processed_queue_depth.fetch_add(1, Ordering::Relaxed);

            // preprocessing: normalize line endings, trim whitespace
            // in actual preprocessing we would preprocess PDFs, images, docx, etc
            let content = file_info.content
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .trim()
                .to_string();
            let _ = handle.block_on(tx.send(ProcessedFile {
                id: file_info.id,
                path: file_info.path,
                content,
            }));
        }
    });
}

fn spawn_chunking_worker(
    handle: tokio::runtime::Handle,
    rx: Arc<Mutex<mpsc::Receiver<ProcessedFile>>>,
    tx: mpsc::Sender<ChunkJob>,
    worker_id: usize,
    processed_queue_depth: Arc<AtomicUsize>,
    chunks_queue_depth: Arc<AtomicUsize>,
) {
    tokio::task::spawn_blocking(move || {
        let model =
            SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
                .create_model()
                .expect("Failed to create tokenizer");

        let target_tokens = MAX_TOKENS_PER_CHUNK;

        loop {
            let maybe_file = handle.block_on(async {
                let mut guard = rx.lock().await;
                guard.recv().await
            });
            let Some(processed_file) = maybe_file else { break };

            println!("Chunker {}: processing {}", worker_id, processed_file.path.display());

            processed_queue_depth.fetch_sub(1, Ordering::Relaxed);

            let mut buf = String::new();
            let mut token_sum = 0usize;
            let mut chunk_idx = 0usize;

            for word in processed_file.content.split_whitespace() {
                let w_tokens = model.get_tokenizer().tokenize(word).len();
                if token_sum + w_tokens > target_tokens && !buf.is_empty() {
                    let chunk_text = std::mem::take(&mut buf);
                    chunks_queue_depth.fetch_add(1, Ordering::Relaxed);
                    let _ = handle.block_on(tx.send(ChunkJob {
                        file_id: processed_file.id.clone(),
                        file_path: processed_file.path.clone(),
                        idx: chunk_idx,
                        text: chunk_text,
                    }));
                    chunk_idx += 1;
                    token_sum = 0;
                }
                if !buf.is_empty() {
                    buf.push(' ');
                }
                buf.push_str(word);
                token_sum += w_tokens;
            }
            if !buf.is_empty() {
                chunks_queue_depth.fetch_add(1, Ordering::Relaxed);
                let _ = handle.block_on(tx.send(ChunkJob {
                    file_id: processed_file.id.clone(),
                    file_path: processed_file.path.clone(),
                    idx: chunk_idx,
                    text: buf,
                }));
            }
        }
    });
}

fn spawn_embedding_worker(
    handle: tokio::runtime::Handle,
    rx: Arc<Mutex<mpsc::Receiver<ChunkJob>>>,
    tx: mpsc::Sender<EmbeddedChunk>,
    worker_id: usize,
    chunks_queue_depth: Arc<AtomicUsize>,
    embeddings_queue_depth: Arc<AtomicUsize>,
) {
    tokio::task::spawn_blocking(move || {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .with_device(tch::Device::Cpu)
            .create_model()
            .expect("model init");

        loop {
            let maybe_job = handle.block_on(async {
                let mut guard = rx.lock().await;
                guard.recv().await
            });
            let Some(job) = maybe_job else { break };

            println!("Embedder {}: processing chunk {} from {}", worker_id, job.idx, job.file_path.display());

            chunks_queue_depth.fetch_sub(1, Ordering::Relaxed);
            embeddings_queue_depth.fetch_add(1, Ordering::Relaxed);

            match model.encode(&[job.text]) {
                Ok(embeddings) => {
                    if let Some(embedding) = embeddings.into_iter().next() {
                        let _ = handle.block_on(tx.send(EmbeddedChunk {
                            file_id: job.file_id,
                            file_path: job.file_path,
                            idx: job.idx,
                            embedding,
                        }));
                    }
                }
                Err(e) => eprintln!("Embedder {} encode failed for chunk {}: {}", worker_id, job.idx, e),
            }
        }
    });
}

/// Counter (final) stage: Consumes the stream and counts results.
/// Receives all processed chunks and returns the total count.
async fn counter_stage(embeddings_rx: Arc<Mutex<mpsc::Receiver<EmbeddedChunk>>>, embeddings_queue_depth: Arc<AtomicUsize>) -> usize {
    let mut count = 0usize;
    while let Some(chunk) = {
        let mut rx = embeddings_rx.lock().await;
        rx.recv().await
    } {
        embeddings_queue_depth.fetch_sub(1, Ordering::Relaxed);

        count += 1;
        println!("completed: {} -> chunk {} (dim={})",
                 chunk.file_path.display(), chunk.idx, chunk.embedding.len());
    }
    println!("Counter: processed {} total chunks", count);
    count
}

async fn monitor_channels(
    paths_depth: Arc<AtomicUsize>,
    content_depth: Arc<AtomicUsize>,
    processed_depth: Arc<AtomicUsize>,
    chunks_depth: Arc<AtomicUsize>,
    embeddings_depth: Arc<AtomicUsize>,
) {
    tokio::time::sleep(Duration::from_millis(10)).await;
    loop {
        let p_depth = paths_depth.load(Ordering::Relaxed);
        let c_depth = content_depth.load(Ordering::Relaxed);
        let pr_depth = processed_depth.load(Ordering::Relaxed);
        let ch_depth = chunks_depth.load(Ordering::Relaxed);
        let e_depth = embeddings_depth.load(Ordering::Relaxed);

        const CYAN: &str = "\x1b[36m";
        const YELLOW: &str = "\x1b[33m";
        const GREEN: &str = "\x1b[32m";
        const RESET: &str = "\x1b[0m";
        const BOLD: &str = "\x1b[1m";

        println!(
            "{bold}{cyan}[Queue Depths]{reset} {yellow}Paths:{p}{reset} → {yellow}Content:{c}{reset} → {yellow}Processed:{pr}{reset} → {yellow}Chunks:{ch}{reset} → {yellow}Embeddings:{e}{reset}",
            bold = BOLD,
            cyan = CYAN,
            yellow = YELLOW,
            reset = RESET,
            p = p_depth,
            c = c_depth,
            pr = pr_depth,
            ch = ch_depth,
            e = e_depth
        );

        if epoch::advance().is_ok() {
            let allocated = stats::allocated::read().unwrap_or(0) as f64 / 1_048_576.0;
            let resident = stats::resident::read().unwrap_or(0) as f64 / 1_048_576.0;
            const MAGENTA: &str = "\x1b[35m";
            println!(
                "{bold}{magenta}[Memory Usage]{reset} {yellow}Allocated:{allocated:.2} MB{reset} | {yellow}Resident:{resident:.2} MB{reset}",
                bold = BOLD,
                magenta = MAGENTA,
                yellow = YELLOW,
                reset = RESET,
                allocated = allocated,
                resident = resident
            );
        }

        if p_depth == 0 && c_depth == 0 && pr_depth == 0 && ch_depth == 0 && e_depth == 0 {
            println!("{bold}{green}[Queue Depths] All queues empty - pipeline complete{reset}", bold = BOLD, green = GREEN, reset = RESET);
            break;
        }

        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn create_test_files(dir: &Path, num_files: usize) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut paths = Vec::new();
    for i in 0..num_files {
        let file_path = dir.join(format!("test_document_{}.txt", i));

        let num_sentences = match i % 5 {
            0 => 2,      // Very short
            1 => 15,     // Short
            2 => 40,     // Medium
            3 => 150,    // Long
            4 => 400,    // Very long
            _ => 50,     // Default medium
        };

        let base_sentence = format!("This is sample text for file {} with varied content. Sentence number X. ", i);
        let content = (0..num_sentences)
            .map(|j| base_sentence.replace("X", &format!("{}", j + 1)))
            .collect::<Vec<_>>()
            .join("");

        let content_len = content.len();
        fs::write(&file_path, &content)?;
        println!("Created test file: {} ({} bytes)", file_path.display(), content_len);
        paths.push(file_path);
    }
    Ok(paths)
}

fn main() -> Result<(), Box<dyn Error>> {
    let reader_runtime = Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("reader-worker")
        .enable_all()
        .build()
        .expect("Failed to build reader runtime");

    let preprocessor_runtime = Arc::new(Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("preprocessor-worker")
        .enable_all()
        .build()
        .expect("Failed to build preprocessor runtime"));

    let chunker_runtime = Arc::new(Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("chunker-worker")
        .enable_all()
        .build()
        .expect("Failed to build chunker runtime"));

    let embedder_runtime = Arc::new(Builder::new_multi_thread()
        .worker_threads(12)
        .thread_name("embedder-worker")
        .enable_all()
        .build()
        .expect("Failed to build embedder runtime"));

    let monitor_runtime = Arc::new(Builder::new_multi_thread()
        .worker_threads(1)
        .thread_name("monitor-worker")
        .enable_all()
        .build()
        .expect("Failed to build monitor runtime"));

    // since run_pipeline includes reader
    let result = reader_runtime.block_on(async {
        run_pipeline(
            preprocessor_runtime.clone(),
            chunker_runtime.clone(),
            embedder_runtime.clone(),
            monitor_runtime.clone(),
        ).await
    });

    drop(reader_runtime);

    result
}

async fn run_pipeline(
    preprocessor_runtime: Arc<tokio::runtime::Runtime>,
    chunker_runtime: Arc<tokio::runtime::Runtime>,
    embedder_runtime: Arc<tokio::runtime::Runtime>,
    monitor_runtime: Arc<tokio::runtime::Runtime>,
) -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("pipeline_usage=info")
        .init();

    info!("Starting pipeline benchmark with separate Tokio runtimes per stage.");

    let num_files = 20;
    let file_readers = 1;
    let preprocessors = 1;
    let chunkers = 2;
    let embedders = 12;

    // for monitoring
    let paths_queue_depth = Arc::new(AtomicUsize::new(0));
    let content_queue_depth = Arc::new(AtomicUsize::new(0));
    let processed_queue_depth = Arc::new(AtomicUsize::new(0));
    let chunks_queue_depth = Arc::new(AtomicUsize::new(0));
    let embeddings_queue_depth = Arc::new(AtomicUsize::new(0));

    let (input_paths_tx, input_paths_rx) = mpsc::channel(DEFAULT_CHANNEL_CAPACITY);
    let (file_content_tx, file_content_rx) = mpsc::channel(DEFAULT_CHANNEL_CAPACITY);
    let (processed_tx, processed_rx) = mpsc::channel(DEFAULT_CHANNEL_CAPACITY);
    let (chunk_job_tx, chunk_job_rx) = mpsc::channel(DEFAULT_CHANNEL_CAPACITY);
    let (embeddings_tx, embeddings_rx) = mpsc::channel(DEFAULT_CHANNEL_CAPACITY);

    let shared_input_paths_rx = Arc::new(Mutex::new(input_paths_rx));
    let shared_file_content_rx = Arc::new(Mutex::new(file_content_rx));
    let shared_processed_rx = Arc::new(Mutex::new(processed_rx));
    let shared_chunk_job_rx = Arc::new(Mutex::new(chunk_job_rx));
    let shared_embeddings_rx = Arc::new(Mutex::new(embeddings_rx));
    
    let pipeline_start_time = Instant::now();

    let monitor_paths_depth = paths_queue_depth.clone();
    let monitor_content_depth = content_queue_depth.clone();
    let monitor_processed_depth = processed_queue_depth.clone();
    let monitor_chunks_depth = chunks_queue_depth.clone();
    let monitor_embeddings_depth = embeddings_queue_depth.clone();

    let monitor_handle = monitor_runtime.spawn(async move {
        monitor_channels(
            monitor_paths_depth,
            monitor_content_depth,
            monitor_processed_depth,
            monitor_chunks_depth,
            monitor_embeddings_depth,
        )
        .await
    });
    
    // File readers -> preprocessors
    for i in 0..file_readers {
        let rx_clone = shared_input_paths_rx.clone();
        let tx_clone = file_content_tx.clone();
        let handle = tokio::runtime::Handle::current();
        spawn_file_reader_worker(handle, rx_clone, tx_clone, i, content_queue_depth.clone(), paths_queue_depth.clone());
    }
    drop(file_content_tx);

    // Preprocessors -> chunkers
    for i in 0..preprocessors {
        let rx_clone = shared_file_content_rx.clone();
        let tx_clone = processed_tx.clone();
        let content_depth = content_queue_depth.clone();
        let processed_depth = processed_queue_depth.clone();

        preprocessor_runtime.spawn(async move {
            spawn_preprocessing_worker(
                tokio::runtime::Handle::current(),
                rx_clone,
                tx_clone,
                i,
                content_depth,
                processed_depth,
            );
        });
    }
    drop(processed_tx);

    // Chunkers -> embedders
    for i in 0..chunkers {
        let rx_clone = shared_processed_rx.clone();
        let tx_clone = chunk_job_tx.clone();
        let processed_depth = processed_queue_depth.clone();
        let chunks_depth = chunks_queue_depth.clone();

        chunker_runtime.spawn(async move {
            spawn_chunking_worker(
                tokio::runtime::Handle::current(),
                rx_clone,
                tx_clone,
                i,
                processed_depth,
                chunks_depth,
            );
        });
    }
    drop(chunk_job_tx);

    for i in 0..embedders {
        let rx_clone = shared_chunk_job_rx.clone();
        let tx_clone = embeddings_tx.clone();
        let chunks_depth = chunks_queue_depth.clone();
        let embeddings_depth = embeddings_queue_depth.clone();

        embedder_runtime.spawn(async move {
            spawn_embedding_worker(
                tokio::runtime::Handle::current(),
                rx_clone,
                tx_clone,
                i,
                chunks_depth,
                embeddings_depth,
            );
        });
    }
    drop(embeddings_tx);
    
    // sink, consumes and returns the total count
    let counter_embeddings_depth = embeddings_queue_depth.clone();
    let counter_handle = tokio::spawn(counter_stage(shared_embeddings_rx.clone(), counter_embeddings_depth));

    let temp_dir = tempfile::tempdir()?;
    let file_paths = create_test_files(temp_dir.path(), num_files)?;

    println!("Created {} test files with varying lengths", num_files);
    let paths_depth_for_enqueue = paths_queue_depth.clone();
    for path in file_paths {
        println!("enqueue: {}", path.display());

        paths_depth_for_enqueue.fetch_add(1, Ordering::Relaxed);
        input_paths_tx.send(path).await?;
    }
    drop(input_paths_tx);

    let results = counter_handle.await?;

    let total_elapsed = pipeline_start_time.elapsed();
    info!(
        "Pipeline finished. Processed {} chunks in {:.2?}.",
        results,
        total_elapsed
    );
    
    println!("finished: {} chunks in {:.2?}", results, total_elapsed);

    let _ = monitor_handle.await;

    Ok(())
}
