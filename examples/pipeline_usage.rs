use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};
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

const NUM_FILES: usize = 20;
const FILE_READERS: usize = 4;
const PREPROCESSORS: usize = 4;
const CHUNKERS: usize = 2;
const EMBEDDERS: usize = 12;


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

async fn file_reader_worker(
    rx: Arc<Mutex<mpsc::Receiver<PathBuf>>>,
    tx: mpsc::Sender<FileInfo>,
    worker_id: usize,
    content_queue_depth: Arc<AtomicUsize>,
    paths_queue_depth: Arc<AtomicUsize>,
) {
    loop {
        let maybe_path = {
            // Acquire an asynchronous lock on the Mutex.
            // If the lock is already held by another worker, this `.await` call will
            // yield, allowing the Tokio scheduler to run other tasks. This worker
            // will be woken up once the lock is available.
            let mut guard = rx.lock().await;

            // Once the lock is acquired, receive a path from the channel.
            // The lock guard is held across this `.await`. This is safe because
            // the `Mutex` is a Tokio-aware Mutex. The guard will be automatically
            // released when it goes out of scope at the end of this block.

            // Q: What happens if worker1 gets the lock but has to wait for a message
            //    (at `guard.recv().await`)? Are other workers blocked?
            // A: Yes, other file_reader workers are "blocked" from accessing this specific
            //    receiver, but in a good way. When worker2 hits `rx.lock().await`,
            //    it doesn't block the OS thread. It yields control to the Tokio
            //    scheduler, which can then run any other ready task (e.g., a
            //    preprocessor or an embedder). The system remains productive.
            //    If guard.recv().await doesn't finish, it means there's nothing
            //    to read yet anyway, so other workers are technically not blocked.
            guard.recv().await
        };
        let Some(path) = maybe_path else { break };

        println!("Reader {}: processing {}", worker_id, path.display());
        paths_queue_depth.fetch_sub(1, Ordering::Relaxed);

        // `tokio::fs` operations are asynchronous. Under the hood, Tokio
        // manages a thread pool dedicated to blocking file system operations.
        // When you call `read_to_string`, the work is sent to one of those
        // threads, and the current async task yields. This frees up the
        // current worker thread to run other async tasks. When the file
        // read is complete, this task is woken up to continue.
        match tokio::fs::read_to_string(&path).await {
            Ok(content) => {
                content_queue_depth.fetch_add(1, Ordering::Relaxed);
                let file_id = Uuid::new_v4().to_string();
                if tx.send(FileInfo { id: file_id, path, content }).await.is_err() {
                    eprintln!("Reader {}: channel closed.", worker_id);
                    break;
                }
            }
            Err(e) => eprintln!("Reader {} error reading file {}: {}", worker_id, path.display(), e),
        }
    }
}

async fn preprocessing_worker(
    rx: Arc<Mutex<mpsc::Receiver<FileInfo>>>,
    tx: mpsc::Sender<ProcessedFile>,
    worker_id: usize,
    content_queue_depth: Arc<AtomicUsize>,
    processed_queue_depth: Arc<AtomicUsize>,
) {
    loop {
        let maybe_file = {
            // This follows the same `lock -> recv` pattern as the file_reader_worker.
            // It allows multiple preprocessor workers to safely share a single
            // input channel.
            let mut guard = rx.lock().await;
            guard.recv().await
        };
        let Some(file_info) = maybe_file else { break };

        println!("Preprocessor {}: processing {}", worker_id, file_info.path.display());

        content_queue_depth.fetch_sub(1, Ordering::Relaxed);
        processed_queue_depth.fetch_add(1, Ordering::Relaxed);

        // --- Async vs. Blocking Bar ---
        // This current preprocessing is just fast string manipulation. It's not
        // worth the overhead of sending it to a blocking thread.
        //
        // However, if this stage involved heavy CPU work (e.g., converting a
        // PDF/DOCX file to text), it would become a CPU-bound task. In that
        // case, we would change this worker to a synchronous `fn` and spawn it
        // with `tokio::task::spawn_blocking` to avoid stalling the async runtime.
        //
        // Rule of thumb: A function is async if it *waits* (I/O, network).
        // It should be on a blocking thread if it *works* (heavy computation).
        // Any pure CPU task taking more than ~100µs is a good candidate for spawn_blocking.
        let content = file_info.content
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .trim()
            .to_string();

        if tx.send(ProcessedFile {
            id: file_info.id,
            path: file_info.path,
            content,
        }).await.is_err() {
            eprintln!("Preprocessor {}: channel closed.", worker_id);
            break;
        }
    }
}

fn chunking_worker(
    rx: Arc<Mutex<mpsc::Receiver<ProcessedFile>>>,
    tx: mpsc::Sender<ChunkJob>,
    worker_id: usize,
    processed_queue_depth: Arc<AtomicUsize>,
    chunks_queue_depth: Arc<AtomicUsize>,
) {
    // This worker is CPU-bound due to the tokenization model.
    // It's intentionally a synchronous function (`fn`) and not `async fn`.
    // It will be run on a dedicated blocking thread using `tokio::task::spawn_blocking`.
    // This prevents it from hogging a core worker thread from the async runtime,
    // which is needed for the I/O-bound tasks to remain responsive.
    let model =
        SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
            .create_model()
            .expect("Failed to create tokenizer");

    let target_tokens = MAX_TOKENS_PER_CHUNK;

    loop {
        // Since this is a synchronous function, we cannot use `.await`.
        // `blocking_lock()` blocks the *current thread* (the spawned blocking
        // thread) until the mutex lock is acquired. This is the correct and
        // efficient behaviour here, as this thread's only job is to wait for
        // work. The lock guard is dropped at the end of the statement,
        // releasing the lock immediately so another chunker can take its turn.
        let maybe_file = rx.blocking_lock().blocking_recv();
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
                if tx.blocking_send(ChunkJob {
                    file_id: processed_file.id.clone(),
                    file_path: processed_file.path.clone(),
                    idx: chunk_idx,
                    text: chunk_text,
                }).is_err() {
                    eprintln!("Chunker {}: channel closed.", worker_id);
                    return;
                }
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
            if tx.blocking_send(ChunkJob {
                file_id: processed_file.id.clone(),
                file_path: processed_file.path.clone(),
                idx: chunk_idx,
                text: buf,
            }).is_err() {
                eprintln!("Chunker {}: channel closed.", worker_id);
                return;
            }
        }
    }
}

fn embedding_worker(
    rx: Arc<Mutex<mpsc::Receiver<ChunkJob>>>,
    tx: mpsc::Sender<EmbeddedChunk>,
    worker_id: usize,
    chunks_queue_depth: Arc<AtomicUsize>,
    embeddings_queue_depth: Arc<AtomicUsize>,
) {
    // This worker is very CPU-bound due to running the embedding model.
    // Like the chunking_worker, it is a synchronous `fn` designed to be run
    // with `tokio::task::spawn_blocking`.
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .with_device(tch::Device::Cpu)
        .create_model()
        .expect("model init");

    loop {
        // This uses the same `blocking_lock` and `blocking_recv` pattern as the
        // chunking worker. It blocks its dedicated thread to wait for a chunk
        // job, processes it, and then loops to wait for the next one.
        let maybe_job = rx.blocking_lock().blocking_recv();
        let Some(job) = maybe_job else { break };

        println!("Embedder {}: processing chunk {} from {}", worker_id, job.idx, job.file_path.display());

        chunks_queue_depth.fetch_sub(1, Ordering::Relaxed);
        embeddings_queue_depth.fetch_add(1, Ordering::Relaxed);

        match model.encode(&[job.text]) {
            Ok(embeddings) => {
                if let Some(embedding) = embeddings.into_iter().next() {
                    if tx.blocking_send(EmbeddedChunk {
                        file_id: job.file_id,
                        file_path: job.file_path,
                        idx: job.idx,
                        embedding,
                    }).is_err() {
                        eprintln!("Embedder {}: channel closed.", worker_id);
                        return;
                    }
                }
            }
            Err(e) => eprintln!("Embedder {} encode failed for chunk {}: {}", worker_id, job.idx, e),
        }
    }
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
        println!("completed: {} -> chunk {} (dim={})", chunk.file_path.display(), chunk.idx, chunk.embedding.len());
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
        std::fs::write(&file_path, &content)?;
        println!("Created test file: {} ({} bytes)", file_path.display(), content_len);
        paths.push(file_path);
    }
    Ok(paths)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt()
        .with_env_filter("pipeline_usage=info")
        .init();

    info!("Starting pipeline usage example");

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

    let monitor_handle = tokio::spawn(async move {
        monitor_channels(
            monitor_paths_depth,
            monitor_content_depth,
            monitor_processed_depth,
            monitor_chunks_depth,
            monitor_embeddings_depth,
        )
        .await
    });
    
    // File readers are I/O-bound, so they are spawned as regular async tasks
    // on the main Tokio runtime.
    for i in 0..FILE_READERS {
        let rx_clone = shared_input_paths_rx.clone();
        let tx_clone = file_content_tx.clone();
        let content_depth = content_queue_depth.clone();
        let paths_depth = paths_queue_depth.clone();
        tokio::spawn(file_reader_worker(rx_clone, tx_clone, i, content_depth, paths_depth));
    }
    drop(file_content_tx);

    // Preprocessors are very fast, so they also run as regular async tasks.
    for i in 0..PREPROCESSORS {
        let rx_clone = shared_file_content_rx.clone();
        let tx_clone = processed_tx.clone();
        let content_depth = content_queue_depth.clone();
        let processed_depth = processed_queue_depth.clone();

        tokio::spawn(preprocessing_worker(
            rx_clone,
            tx_clone,
            i,
            content_depth,
            processed_depth,
        ));
    }
    drop(processed_tx);

    // Chunkers are CPU-bound, so they are spawned on Tokio's blocking thread pool.
    // This prevents them from blocking the main runtime, keeping I/O tasks responsive.
    for i in 0..CHUNKERS {
        let rx_clone = shared_processed_rx.clone();
        let tx_clone = chunk_job_tx.clone();
        let processed_depth = processed_queue_depth.clone();
        let chunks_depth = chunks_queue_depth.clone();

        tokio::task::spawn_blocking(move || {
            chunking_worker(
                rx_clone,
                tx_clone,
                i,
                processed_depth,
                chunks_depth,
            );
        });
    }
    drop(chunk_job_tx);

    // Embedders are very CPU-bound and are also spawned on the blocking thread pool.
    for i in 0..EMBEDDERS {
        let rx_clone = shared_chunk_job_rx.clone();
        let tx_clone = embeddings_tx.clone();
        let chunks_depth = chunks_queue_depth.clone();
        let embeddings_depth = embeddings_queue_depth.clone();

        tokio::task::spawn_blocking(move || {
            embedding_worker(
                rx_clone,
                tx_clone,
                i,
                chunks_depth,
                embeddings_depth,
            );
        });
    }
    drop(embeddings_tx);
    
    // The final counter stage is a lightweight async task.
    let counter_embeddings_depth = embeddings_queue_depth.clone();
    let counter_handle = tokio::spawn(counter_stage(shared_embeddings_rx.clone(), counter_embeddings_depth));

    let temp_dir = tempfile::tempdir()?;
    let file_paths = create_test_files(temp_dir.path(), NUM_FILES)?;

    println!("Created {} test files with varying lengths", NUM_FILES);
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
