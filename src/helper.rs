// TODO: When embedder is implemented, we need to remove create_mock_embedding

use std::path::{Path, PathBuf};
use std::time;
use std::os::unix::fs::PermissionsExt;
use lancedb::Error;
use arrow_schema::ArrowError;
use crate::store::{AddFile, FileType, LanceStore};

pub fn create_mock_embedding(content: &str, embedding_type: &str) -> Vec<f32> {
    let mut embedding = vec![0.0; 384];

    let hash = content.len() as u32 + content.chars().next().unwrap_or('a') as u32;

    for i in 0..384 {
        embedding[i] = (hash + i as u32) as f32 / 1000.0;
    }

    match embedding_type {
        "filename" => {
            if content.contains(".pdf") {
                embedding[0] += 0.1;
            }
            if content.contains(".txt") {
                embedding[1] += 0.1;
            }
            if content.contains(".md") {
                embedding[2] += 0.1;
            }
        }
        "content" => {
            if content.contains("AI") || content.contains("artificial intelligence") {
                embedding[10] += 0.2;
            }
            if content.contains("cooking") || content.contains("recipe") {
                embedding[11] += 0.2;
            }
            if content.contains("research") || content.contains("study") {
                embedding[12] += 0.2;
            }
        }
        _ => {}
    }

    embedding
}

pub fn compute_md5_hash(content: &str) -> String {
    use md5;
    let digest = md5::compute(content);
    format!("{:x}", digest)
}

pub async fn add_directory(store: &LanceStore, dir_path: &PathBuf, filename_embedding: Vec<f32>) -> Result<String, Error> {
    let path = Path::new(dir_path);
    if !path.exists() || !path.is_dir() {
        return Err(Error::from(ArrowError::InvalidArgumentError(
            format!("Directory does not exist or is not a directory: {}", dir_path.display()),
        )));
    }

    let metadata_fs = path.metadata().map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to get directory metadata: {}", e),
        ))
    })?;
    let modified_time = metadata_fs.modified().map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to get modified time: {}", e),
        ))
    })?.duration_since(time::UNIX_EPOCH).map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to convert time: {}", e),
        ))
    })?.as_secs() as i64;
    let accessed_time = metadata_fs.accessed().map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to get accessed time: {}", e),
        ))
    })?.duration_since(time::UNIX_EPOCH).map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to convert time: {}", e),
        ))
    })?.as_secs() as i64;
    let created_time = metadata_fs.created().map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to get created time: {}", e),
        ))
    })?.duration_since(time::UNIX_EPOCH).map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to convert time: {}", e),
        ))
    })?.as_secs() as i64;
    let permissions = metadata_fs.permissions().mode();

    let params = AddFile::new(dir_path.to_path_buf(), FileType::Directory, filename_embedding)
        .permissions(permissions)
        .file_size(0)
        .modified_time(modified_time)
        .accessed_time(accessed_time)
        .created_time(created_time);

    store.add_file(&params, &store.text_content_embeddings_table).await
}

pub async fn scan_directory(store: &LanceStore, dir_path: &PathBuf) -> Result<Vec<String>, Error> {
    let path = Path::new(dir_path);
    if !path.exists() || !path.is_dir() {
        return Err(Error::from(ArrowError::InvalidArgumentError(
            format!("Directory does not exist or is not a directory: {}", dir_path.display()),
        )));
    }

    let mut file_ids = Vec::new();
    let entries = std::fs::read_dir(path).map_err(|e| {
        Error::from(ArrowError::InvalidArgumentError(
            format!("Failed to read directory: {}", e),
        ))
    })?;

    for entry in entries {
        let entry = entry.map_err(|e| {
            Error::from(ArrowError::InvalidArgumentError(
                format!("Failed to read directory entry: {}", e),
            ))
        })?;
        let path = entry.path();
        
        if path.is_file() {
            let filename_embedding = create_mock_embedding(&path.file_name().unwrap().to_string_lossy(), "filename");
            
            let metadata_fs = path.metadata().map_err(|e| {
                Error::from(ArrowError::InvalidArgumentError(
                    format!("Failed to get file metadata: {}", e),
                ))
            })?;
            let modified_time = metadata_fs.modified().map_err(|e| {
                Error::from(ArrowError::InvalidArgumentError(
                    format!("Failed to get modified time: {}", e),
                ))
            })?.duration_since(time::UNIX_EPOCH).map_err(|e| {
                Error::from(ArrowError::InvalidArgumentError(
                    format!("Failed to convert time: {}", e),
                ))
            })?.as_secs() as i64;
            let accessed_time = metadata_fs.accessed().map_err(|e| {
                Error::from(ArrowError::InvalidArgumentError(
                    format!("Failed to get accessed time: {}", e),
                ))
            })?.duration_since(time::UNIX_EPOCH).map_err(|e| {
                Error::from(ArrowError::InvalidArgumentError(
                    format!("Failed to convert time: {}", e),
                ))
            })?.as_secs() as i64;
            let created_time = metadata_fs.created().map_err(|e| {
                Error::from(ArrowError::InvalidArgumentError(
                    format!("Failed to get created time: {}", e),
                ))
            })?.duration_since(time::UNIX_EPOCH).map_err(|e| {
                Error::from(ArrowError::InvalidArgumentError(
                    format!("Failed to convert time: {}", e),
                ))
            })?.as_secs() as i64;
            let permissions = metadata_fs.permissions().mode();
            let file_size = metadata_fs.len();

            let params = AddFile::new(path.clone(), FileType::File, filename_embedding)
                .permissions(permissions)
                .file_size(file_size)
                .modified_time(modified_time)
                .accessed_time(accessed_time)
                .created_time(created_time);

            let file_id = store.add_text_file(&params).await?;
            file_ids.push(file_id);
        }
    }

    Ok(file_ids)
}
