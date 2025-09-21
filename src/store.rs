use arrow_array::builder::{FixedSizeListBuilder, Float32Builder};
use arrow_array::{
    Array, BinaryArray, Int64Array, RecordBatch, RecordBatchIterator, StringArray, UInt32Array,
    UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use chrono::Utc;
use futures::TryStreamExt;
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{Connection, Result, Table, connect};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time;
use uuid::Uuid;

use crate::types::FilePath;

const METADATA_TABLE_NAME: &str = "file_metadata";
const FILENAME_EMBEDDINGS_TABLE_NAME: &str = "filename_embeddings";
const TEXT_CONTENT_EMBEDDINGS_TABLE_NAME: &str = "text_content_embeddings";
const IMAGE_CONTENT_EMBEDDINGS_TABLE_NAME: &str = "image_content_embeddings";
const EMBEDDING_DIMENSION: i32 = 384;

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("Column not found: {0}")]
    ColumnNotFound(String),
    #[error("Invalid column type for {0}")]
    InvalidColumnType(String),
    #[error("Row index out of bounds: {0}")]
    RowIndexOutOfBounds(usize),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("LanceDB error: {0}")]
    LanceDB(#[from] lancedb::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FileType {
    File,
    Directory,
    Symlink,
}

impl FileType {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileType::File => "file",
            FileType::Directory => "directory",
            FileType::Symlink => "symlink",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "file" => Some(FileType::File),
            "directory" => Some(FileType::Directory),
            "symlink" => Some(FileType::Symlink),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileMetadata {
    // All the non-optional fields can be accessed via std::fs and exists on every file system
    pub id: String, // not sure why this is needed, but keeping it for now
    // NOTE that we don't need file_extension because we can use "LIKE %.ext" to search for files with a specific extension
    // we also don't need file_name because we can use "LIKE %name%" to search for files with a specific name
    // FilePath handles cross-platform path encoding safely
    pub file_path: FilePath,
    pub file_path_lossy: String, // Lossy string representation for searching
    pub file_type: FileType, // are there any other file types other than file, directory, and symlink?
    pub mime_type: Option<String>, // MIME type from file extension
    pub permissions: Option<u32>, // Unix permissions
    pub file_size: u64,
    pub modified_time: i64,         // Last modified timestamp
    pub accessed_time: Option<i64>, // Last accessed timestamp (OS-dependent)
    pub created_time: Option<i64>,  // Creation timestamp (OS-dependent)
    pub tags: Option<Vec<String>>,  //  macOS tags? planning to use this for labels
}

impl FileMetadata {
    pub fn filename(&self) -> Option<String> {
        let os_string = FilePath::decode_to_os_string(&self.file_path.bytes);

        PathBuf::from(os_string)
            .file_name()
            .and_then(|name| name.to_str())
            .map(|s| s.to_string())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileNameEmbedding {
    pub metadata_id: String,
    pub filename_embedding: Vec<f32>,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextContentEmbedding {
    pub metadata_id: String,
    pub content_embedding: Vec<f32>,
    pub content_hash: String,
    pub created_at: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContentEmbedding {
    pub metadata_id: String,
    pub content_embedding: Vec<f32>,
    pub content_hash: String,
    pub created_at: i64,
}

#[derive(Debug, Clone)]
pub struct AddFile {
    pub file_path: FilePath,
    pub file_path_lossy: String, // Lossy string representation for searching
    pub file_type: FileType,
    pub mime_type: Option<String>, // MIME type
    pub permissions: Option<u32>,  // Unix permissions
    pub file_size: u64,
    pub modified_time: i64,
    pub accessed_time: Option<i64>,
    pub created_time: Option<i64>,
    pub tags: Option<Vec<String>>,
    pub filename_embedding: Vec<f32>,
    pub content_embedding: Option<Vec<f32>>,
    pub content_hash: Option<String>,
}

impl AddFile {
    pub fn new(file_path: PathBuf, file_type: FileType, filename_embedding: Vec<f32>) -> Self {
        let now = chrono::Utc::now().timestamp();
        let file_path_lossy = file_path.to_string_lossy().to_string();

        Self {
            file_path: FilePath::from(file_path),
            file_path_lossy,
            file_type,
            mime_type: None,
            permissions: None,
            file_size: 0,
            modified_time: now,
            accessed_time: Some(now),
            created_time: Some(now),
            tags: None,
            filename_embedding,
            content_embedding: None,
            content_hash: None,
        }
    }

    /// If you're using this outside of the tests, you probably want to use the builder pattern instead
    pub fn override_file_path(mut self, file_path: PathBuf) -> Self {
        self.file_path = FilePath::from(file_path.clone());
        self.file_path_lossy = file_path.to_string_lossy().to_string();
        self
    }

    pub fn mime_type(mut self, mime_type: &str) -> Self {
        self.mime_type = Some(mime_type.to_string());
        self
    }

    pub fn permissions(mut self, permissions: u32) -> Self {
        self.permissions = Some(permissions);
        self
    }

    pub fn file_size(mut self, file_size: u64) -> Self {
        self.file_size = file_size;
        self
    }

    pub fn modified_time(mut self, modified_time: i64) -> Self {
        self.modified_time = modified_time;
        self
    }

    pub fn accessed_time(mut self, accessed_time: i64) -> Self {
        self.accessed_time = Some(accessed_time);
        self
    }

    pub fn created_time(mut self, created_time: i64) -> Self {
        self.created_time = Some(created_time);
        self
    }

    pub fn tags(mut self, tags: &[&str]) -> Self {
        if tags.is_empty() {
            self.tags = None;
        } else {
            self.tags = Some(tags.iter().map(|s| s.to_string()).collect());
        }
        self
    }

    pub fn content_embedding(mut self, content_embedding: Vec<f32>) -> Self {
        self.content_embedding = Some(content_embedding);
        self
    }

    pub fn content_hash(mut self, content_hash: &str) -> Self {
        self.content_hash = Some(content_hash.to_string());
        self
    }
}

#[derive(Debug, Clone)]
pub struct MetadataUpdate {
    pub file_path: Option<FilePath>,
    pub file_path_lossy: Option<String>, // Lossy string representation for searching
    pub mime_type: Option<String>,
    pub permissions: Option<u32>,
    pub file_size: Option<u64>,
    pub modified_time: Option<i64>,
    pub accessed_time: Option<i64>,
    pub created_time: Option<i64>,
    pub tags: Option<Vec<String>>,
}

impl MetadataUpdate {
    pub fn new() -> Self {
        Self {
            file_path: None,
            file_path_lossy: None,
            mime_type: None,
            permissions: None,
            file_size: None,
            modified_time: None,
            accessed_time: None,
            created_time: None,
            tags: None,
        }
    }

    pub fn file_path(mut self, file_path: PathBuf) -> Self {
        // note: updates both file_path and file_path_lossy
        let file_path_lossy = file_path.to_string_lossy().to_string();
        self.file_path = Some(FilePath::from(file_path));
        self.file_path_lossy = Some(file_path_lossy);
        self
    }

    pub fn file_path_lossy(mut self, file_path_lossy: String) -> Self {
        // shouldn't be needed, but keeping it for now
        self.file_path_lossy = Some(file_path_lossy);
        self
    }

    pub fn mime_type(mut self, mime_type: &str) -> Self {
        self.mime_type = Some(mime_type.to_string());
        self
    }

    pub fn permissions(mut self, permissions: u32) -> Self {
        self.permissions = Some(permissions);
        self
    }

    pub fn file_size(mut self, file_size: u64) -> Self {
        self.file_size = Some(file_size);
        self
    }

    pub fn modified_time(mut self, modified_time: i64) -> Self {
        self.modified_time = Some(modified_time);
        self
    }

    pub fn accessed_time(mut self, accessed_time: i64) -> Self {
        self.accessed_time = Some(accessed_time);
        self
    }

    pub fn created_time(mut self, created_time: i64) -> Self {
        self.created_time = Some(created_time);
        self
    }

    pub fn tags(mut self, tags: &[&str]) -> Self {
        if tags.is_empty() {
            self.tags = None;
        } else {
            self.tags = Some(tags.iter().map(|s| s.to_string()).collect());
        }
        self
    }
}

#[derive(Clone)]
pub struct LanceStore {
    pub metadata_table: Table,
    pub filename_embeddings_table: Table,
    pub text_content_embeddings_table: Table,
    pub image_content_embeddings_table: Table,
}

impl Debug for LanceStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LanceStore")
            .field("metadata_table", &"<Table>")
            .field("filename_embeddings_table", &"<Table>")
            .field("text_content_embeddings_table", &"<Table>")
            .field("image_content_embeddings_table", &"<Table>")
            .finish()
    }
}

impl LanceStore {
    pub async fn new(db_uri: &str) -> Result<Self> {
        let db = connect(db_uri).execute().await?;

        let metadata_table = Self::get_or_create_metadata_table(&db).await?;
        let filename_embeddings_table = Self::get_or_create_filename_embeddings_table(&db).await?;
        let text_content_embeddings_table =
            Self::get_or_create_text_content_embeddings_table(&db).await?;
        let image_content_embeddings_table =
            Self::get_or_create_image_content_embeddings_table(&db).await?;

        Ok(Self {
            metadata_table,
            filename_embeddings_table,
            text_content_embeddings_table,
            image_content_embeddings_table,
        })
    }

    pub fn extract_metadata_from_batch(
        batch: &RecordBatch,
        row: usize,
    ) -> std::result::Result<FileMetadata, StoreError> {
        if row >= batch.num_rows() {
            return Err(StoreError::RowIndexOutOfBounds(row));
        }

        let id_col = batch
            .column_by_name("id")
            .ok_or_else(|| StoreError::ColumnNotFound("id".to_string()))?;
        let file_path_col = batch
            .column_by_name("file_path")
            .ok_or_else(|| StoreError::ColumnNotFound("file_path".to_string()))?;
        let file_path_lossy_col = batch
            .column_by_name("file_path_lossy")
            .ok_or_else(|| StoreError::ColumnNotFound("file_path_lossy".to_string()))?;
        let file_type_col = batch
            .column_by_name("file_type")
            .ok_or_else(|| StoreError::ColumnNotFound("file_type".to_string()))?;
        let mime_type_col = batch
            .column_by_name("mime_type")
            .ok_or_else(|| StoreError::ColumnNotFound("mime_type".to_string()))?;
        let permissions_col = batch
            .column_by_name("permissions")
            .ok_or_else(|| StoreError::ColumnNotFound("permissions".to_string()))?;
        let file_size_col = batch
            .column_by_name("file_size")
            .ok_or_else(|| StoreError::ColumnNotFound("file_size".to_string()))?;
        let modified_time_col = batch
            .column_by_name("modified_time")
            .ok_or_else(|| StoreError::ColumnNotFound("modified_time".to_string()))?;
        let accessed_time_col = batch
            .column_by_name("accessed_time")
            .ok_or_else(|| StoreError::ColumnNotFound("accessed_time".to_string()))?;
        let created_time_col = batch
            .column_by_name("created_time")
            .ok_or_else(|| StoreError::ColumnNotFound("created_time".to_string()))?;
        let tags_col = batch
            .column_by_name("tags")
            .ok_or_else(|| StoreError::ColumnNotFound("tags".to_string()))?;

        let id = id_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StoreError::InvalidColumnType("id".to_string()))?
            .value(row);
        let file_path_bytes = file_path_col
            .as_any()
            .downcast_ref::<BinaryArray>()
            .ok_or_else(|| StoreError::InvalidColumnType("file_path".to_string()))?
            .value(row);
        let file_path = FilePath::from_bytes(file_path_bytes.to_vec());
        let file_path_lossy = file_path_lossy_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StoreError::InvalidColumnType("file_path_lossy".to_string()))?
            .value(row);
        let file_type_str = file_type_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StoreError::InvalidColumnType("file_type".to_string()))?
            .value(row);
        let mime_type_array = mime_type_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StoreError::InvalidColumnType("mime_type".to_string()))?;
        let mime_type = if mime_type_array.is_null(row) {
            None
        } else {
            Some(mime_type_array.value(row).to_string())
        };

        let permissions_array = permissions_col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| StoreError::InvalidColumnType("permissions".to_string()))?;
        let permissions = if permissions_array.is_null(row) {
            None
        } else {
            Some(permissions_array.value(row))
        };

        let file_size = file_size_col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| StoreError::InvalidColumnType("file_size".to_string()))?
            .value(row);
        let modified_time = modified_time_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StoreError::InvalidColumnType("modified_time".to_string()))?
            .value(row);

        let accessed_time_array = accessed_time_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StoreError::InvalidColumnType("accessed_time".to_string()))?;
        let accessed_time = if accessed_time_array.is_null(row) {
            None
        } else {
            Some(accessed_time_array.value(row))
        };

        let created_time_array = created_time_col
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StoreError::InvalidColumnType("created_time".to_string()))?;
        let created_time = if created_time_array.is_null(row) {
            None
        } else {
            Some(created_time_array.value(row))
        };

        let tags_array = tags_col
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| StoreError::InvalidColumnType("tags".to_string()))?;
        let tags_str = if tags_array.is_null(row) {
            None
        } else {
            Some(tags_array.value(row))
        };

        let file_type = FileType::from_str(file_type_str)
            .ok_or_else(|| StoreError::InvalidColumnType("file_type".to_string()))?;

        Ok(FileMetadata {
            id: id.to_string(),
            file_path,
            file_path_lossy: file_path_lossy.to_string(),
            file_type,
            mime_type: mime_type.map(|s| s.to_string()),
            permissions,
            file_size,
            modified_time,
            accessed_time,
            created_time,
            tags: tags_str
                .map(|s| {
                    if s.is_empty() {
                        None
                    } else {
                        Some(
                            s.split(',')
                                .map(|tag| tag.trim().to_string())
                                .collect::<Vec<String>>(),
                        )
                    }
                })
                .flatten(),
        })
    }

    async fn get_or_create_metadata_table(db: &Connection) -> Result<Table> {
        if db
            .table_names()
            .execute()
            .await?
            .contains(&METADATA_TABLE_NAME.to_string())
        {
            db.open_table(METADATA_TABLE_NAME).execute().await
        } else {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Utf8, false),
                Field::new("file_path", DataType::Binary, false),
                Field::new("file_path_lossy", DataType::Utf8, false),
                Field::new("file_type", DataType::Utf8, false),
                Field::new("mime_type", DataType::Utf8, true),
                Field::new("permissions", DataType::UInt32, true),
                Field::new("file_size", DataType::UInt64, false),
                Field::new("modified_time", DataType::Int64, false),
                Field::new("accessed_time", DataType::Int64, true),
                Field::new("created_time", DataType::Int64, true),
                Field::new("tags", DataType::Utf8, true),
            ]));

            let empty_batch = RecordBatch::new_empty(schema.clone());
            let batches = RecordBatchIterator::new(vec![Ok(empty_batch)].into_iter(), schema);
            db.create_table(METADATA_TABLE_NAME, Box::new(batches))
                .execute()
                .await
        }
    }

    async fn get_or_create_filename_embeddings_table(db: &Connection) -> Result<Table> {
        if db
            .table_names()
            .execute()
            .await?
            .contains(&FILENAME_EMBEDDINGS_TABLE_NAME.to_string())
        {
            db.open_table(FILENAME_EMBEDDINGS_TABLE_NAME)
                .execute()
                .await
        } else {
            let schema = Arc::new(Schema::new(vec![
                Field::new("metadata_id", DataType::Utf8, false),
                Field::new(
                    "filename_embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        EMBEDDING_DIMENSION,
                    ),
                    false,
                ),
                Field::new("created_at", DataType::Int64, false),
            ]));

            let empty_batch = RecordBatch::new_empty(schema.clone());
            let batches = RecordBatchIterator::new(vec![Ok(empty_batch)].into_iter(), schema);
            db.create_table(FILENAME_EMBEDDINGS_TABLE_NAME, Box::new(batches))
                .execute()
                .await
        }
    }

    async fn get_or_create_text_content_embeddings_table(db: &Connection) -> Result<Table> {
        if db
            .table_names()
            .execute()
            .await?
            .contains(&TEXT_CONTENT_EMBEDDINGS_TABLE_NAME.to_string())
        {
            db.open_table(TEXT_CONTENT_EMBEDDINGS_TABLE_NAME)
                .execute()
                .await
        } else {
            let schema = Arc::new(Schema::new(vec![
                Field::new("metadata_id", DataType::Utf8, false),
                Field::new(
                    "content_embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        EMBEDDING_DIMENSION,
                    ),
                    false,
                ),
                Field::new("content_hash", DataType::Utf8, false),
                Field::new("created_at", DataType::Int64, false),
            ]));

            let empty_batch = RecordBatch::new_empty(schema.clone());
            let batches = RecordBatchIterator::new(vec![Ok(empty_batch)].into_iter(), schema);
            db.create_table(TEXT_CONTENT_EMBEDDINGS_TABLE_NAME, Box::new(batches))
                .execute()
                .await
        }
    }

    async fn get_or_create_image_content_embeddings_table(db: &Connection) -> Result<Table> {
        if db
            .table_names()
            .execute()
            .await?
            .contains(&IMAGE_CONTENT_EMBEDDINGS_TABLE_NAME.to_string())
        {
            db.open_table(IMAGE_CONTENT_EMBEDDINGS_TABLE_NAME)
                .execute()
                .await
        } else {
            let schema = Arc::new(Schema::new(vec![
                Field::new("metadata_id", DataType::Utf8, false),
                Field::new(
                    "content_embedding",
                    DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        EMBEDDING_DIMENSION,
                    ),
                    false,
                ),
                Field::new("content_hash", DataType::Utf8, false),
                Field::new("created_at", DataType::Int64, false),
            ]));

            let empty_batch = RecordBatch::new_empty(schema.clone());
            let batches = RecordBatchIterator::new(vec![Ok(empty_batch)].into_iter(), schema);
            db.create_table(IMAGE_CONTENT_EMBEDDINGS_TABLE_NAME, Box::new(batches))
                .execute()
                .await
        }
    }
}

impl LanceStore {
    pub async fn get_metadata(&self, id: &str) -> Result<Option<FileMetadata>> {
        let query = self
            .metadata_table
            .query()
            .only_if(format!("id = '{}'", id));

        let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

        if let Some(batch) = batches.first() {
            if batch.num_rows() > 0 {
                match Self::extract_metadata_from_batch(batch, 0) {
                    Ok(metadata) => return Ok(Some(metadata)),
                    Err(e) => {
                        return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                            format!("Failed to extract metadata: {}", e),
                        )));
                    }
                }
            }
        }
        Ok(None)
    }

    pub async fn get_metadata_by_path(&self, file_path: &PathBuf) -> Result<Option<FileMetadata>> {
        let file_path_binary = FilePath::from(file_path.clone());

        let hex_string = file_path_binary
            .bytes
            .iter()
            .map(|b| format!("{:02x}", b))
            .collect::<String>();

        let binary_query = format!("file_path = X'{}'", hex_string);
        let query = self
            .metadata_table
            .query()
            // .only_if(format!("file_path_lossy = '{}'", file_path.to_string_lossy()));
            .only_if(binary_query);

        let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

        if let Some(batch) = batches.first() {
            if batch.num_rows() > 0 {
                match Self::extract_metadata_from_batch(batch, 0) {
                    Ok(metadata) => return Ok(Some(metadata)),
                    Err(e) => {
                        return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                            format!("Failed to extract metadata: {}", e),
                        )));
                    }
                }
            }
        }
        Ok(None)
    }

    pub async fn add_text_file(&self, params: &AddFile) -> Result<String> {
        self.add_file(params, &self.text_content_embeddings_table)
            .await
    }

    pub async fn add_image_file(&self, params: &AddFile) -> Result<String> {
        self.add_file(params, &self.image_content_embeddings_table)
            .await
    }

    pub async fn move_file(&self, old_path: &PathBuf, new_path: PathBuf) -> Result<()> {
        let metadata = self.get_metadata_by_path(old_path).await?;
        if metadata.is_none() {
            return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                format!("File not found at path: {}", old_path.display()),
            )));
        }

        let metadata = metadata.expect("Unreachable");
        let updates = MetadataUpdate::new()
            .file_path(new_path)
            .modified_time(chrono::Utc::now().timestamp());

        self.update_metadata(&metadata.id, &updates).await
    }

    pub async fn update_text_file_content(
        &self,
        file_path: &PathBuf,
        new_content: &str,
        new_content_embedding: Vec<f32>,
        content_hash: &str,
    ) -> Result<()> {
        let metadata = self.get_metadata_by_path(file_path).await?;
        if metadata.is_none() {
            return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                format!("File not found at path: {}", file_path.display()),
            )));
        }

        let metadata = metadata.unwrap();
        let now = chrono::Utc::now().timestamp();

        let content_schema = self.text_content_embeddings_table.schema().await?;
        let mut content_builder =
            FixedSizeListBuilder::new(Float32Builder::new(), new_content_embedding.len() as i32);
        content_builder
            .values()
            .append_slice(&new_content_embedding);
        content_builder.append(true);
        let content_array = content_builder.finish();

        let content_batch = RecordBatch::try_new(
            content_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![metadata.id.clone()])),
                Arc::new(content_array),
                Arc::new(StringArray::from(vec![content_hash])),
                Arc::new(Int64Array::from(vec![now])),
            ],
        )?;

        self.text_content_embeddings_table
            .delete(&format!("metadata_id = '{}'", metadata.id))
            .await?;

        let batches = RecordBatchIterator::new(vec![Ok(content_batch)].into_iter(), content_schema);
        let add_result = self
            .text_content_embeddings_table
            .add(Box::new(batches))
            .execute()
            .await;

        if let Err(e) = add_result {
            eprintln!("Failed to add new content embedding: {:?}", e);
            return Err(e);
        }

        let updates = MetadataUpdate::new()
            .file_size(new_content.len() as u64)
            .modified_time(now)
            .accessed_time(now);

        self.update_metadata(&metadata.id, &updates).await
    }

    pub async fn update_image_file_content(
        &self,
        file_path: &PathBuf,
        new_content: &[u8],
        new_content_embedding: Vec<f32>,
        content_hash: &str,
    ) -> Result<()> {
        let metadata = self.get_metadata_by_path(file_path).await?;
        if metadata.is_none() {
            return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                format!("File not found at path: {}", file_path.display()),
            )));
        }

        let metadata = metadata.unwrap();
        let now = chrono::Utc::now().timestamp();

        let content_schema = self.image_content_embeddings_table.schema().await?;
        let mut content_builder =
            FixedSizeListBuilder::new(Float32Builder::new(), new_content_embedding.len() as i32);
        content_builder
            .values()
            .append_slice(&new_content_embedding);
        content_builder.append(true);
        let content_array = content_builder.finish();

        let content_batch = RecordBatch::try_new(
            content_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![metadata.id.clone()])),
                Arc::new(content_array),
                Arc::new(StringArray::from(vec![content_hash])),
                Arc::new(Int64Array::from(vec![now])),
            ],
        )?;

        self.image_content_embeddings_table
            .delete(&format!("metadata_id = '{}'", metadata.id))
            .await?;

        let batches = RecordBatchIterator::new(vec![Ok(content_batch)].into_iter(), content_schema);
        self.image_content_embeddings_table
            .add(Box::new(batches))
            .execute()
            .await?;

        let updates = MetadataUpdate::new()
            .file_size(new_content.len() as u64)
            .modified_time(now)
            .accessed_time(now);

        self.update_metadata(&metadata.id, &updates).await
    }

    pub async fn delete_file(&self, file_path: &PathBuf) -> Result<()> {
        let metadata = self.get_metadata_by_path(file_path).await?;
        if metadata.is_none() {
            return Ok(());
        }

        let metadata = metadata.unwrap();
        self.delete_file_by_metadata_id(&metadata.id).await
    }

    pub async fn reindex_file(&self, file_path: &PathBuf) -> Result<()> {
        let metadata = self.get_metadata_by_path(file_path).await?;
        if metadata.is_none() {
            return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                format!("File not found at path: {}", file_path.display()),
            )));
        }

        let metadata = metadata.unwrap();

        let os_string = FilePath::decode_to_os_string(&metadata.file_path.bytes);
        let path = PathBuf::from(os_string);

        if !path.exists() {
            self.delete_file_by_metadata_id(&metadata.id).await?;
            return Ok(());
        }

        let metadata_fs = path.metadata().map_err(|e| {
            lancedb::Error::from(ArrowError::InvalidArgumentError(format!(
                "Failed to get file metadata: {}",
                e
            )))
        })?;
        let modified_time = metadata_fs
            .modified()
            .map_err(|e| {
                lancedb::Error::from(ArrowError::InvalidArgumentError(format!(
                    "Failed to get modified time: {}",
                    e
                )))
            })?
            .duration_since(time::UNIX_EPOCH)
            .map_err(|e| {
                lancedb::Error::from(ArrowError::InvalidArgumentError(format!(
                    "Failed to convert time: {}",
                    e
                )))
            })?
            .as_secs() as i64;
        let accessed_time = metadata_fs
            .accessed()
            .map_err(|e| {
                lancedb::Error::from(ArrowError::InvalidArgumentError(format!(
                    "Failed to get accessed time: {}",
                    e
                )))
            })?
            .duration_since(time::UNIX_EPOCH)
            .map_err(|e| {
                lancedb::Error::from(ArrowError::InvalidArgumentError(format!(
                    "Failed to convert time: {}",
                    e
                )))
            })?
            .as_secs() as i64;
        let created_time = metadata_fs
            .created()
            .map_err(|e| {
                lancedb::Error::from(ArrowError::InvalidArgumentError(format!(
                    "Failed to get created time: {}",
                    e
                )))
            })?
            .duration_since(time::UNIX_EPOCH)
            .map_err(|e| {
                lancedb::Error::from(ArrowError::InvalidArgumentError(format!(
                    "Failed to convert time: {}",
                    e
                )))
            })?
            .as_secs() as i64;
        let file_size = metadata_fs.len();
        let permissions = metadata_fs.permissions().mode();

        let updates = MetadataUpdate::new()
            .permissions(permissions)
            .file_size(file_size)
            .modified_time(modified_time)
            .accessed_time(accessed_time)
            .created_time(created_time);

        self.update_metadata(&metadata.id, &updates).await
    }

    pub async fn update_tags(&self, file_path: &PathBuf, new_tags: Vec<String>) -> Result<()> {
        let metadata = self.get_metadata_by_path(file_path).await?;
        if metadata.is_none() {
            return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                format!("File not found at path: {}", file_path.display()),
            )));
        }

        let metadata = metadata.expect("Unreachable");
        let tag_refs: Vec<&str> = new_tags.iter().map(|s| s.as_str()).collect();
        let updates = MetadataUpdate::new().tags(&tag_refs);

        self.update_metadata(&metadata.id, &updates).await
    }

    async fn update_metadata(&self, id: &str, updates: &MetadataUpdate) -> Result<()> {
        // CURRENT WAY: we delete the metadata and add it back with the new values
        // maybe update metadata in place?
        // one other thing is that we can't set anything to None, because why would we ever do that?

        let current_metadata = self.get_metadata(id).await?;
        if current_metadata.is_none() {
            return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                format!("Metadata with id '{}' not found", id),
            )));
        }

        let current = current_metadata.unwrap();

        let mut new_metadata = current.clone();

        if let Some(file_path) = &updates.file_path {
            new_metadata.file_path = file_path.clone();
            let os_string = FilePath::decode_to_os_string(&file_path.bytes);
            new_metadata.file_path_lossy = os_string.to_string_lossy().to_string();
        }
        if let Some(file_path_lossy) = &updates.file_path_lossy {
            // shouldn't be needed, but keeping it for now
            new_metadata.file_path_lossy = file_path_lossy.clone();
        }
        if let Some(mime_type) = &updates.mime_type {
            new_metadata.mime_type = Some(mime_type.clone());
        }
        if let Some(permissions) = updates.permissions {
            new_metadata.permissions = Some(permissions);
        }
        if let Some(file_size) = updates.file_size {
            new_metadata.file_size = file_size;
        }
        if let Some(modified_time) = updates.modified_time {
            new_metadata.modified_time = modified_time;
        }
        if let Some(accessed_time) = updates.accessed_time {
            new_metadata.accessed_time = Some(accessed_time);
        }
        if let Some(created_time) = updates.created_time {
            new_metadata.created_time = Some(created_time);
        }
        if let Some(tags) = &updates.tags {
            new_metadata.tags = Some(tags.clone());
        }

        self.delete_metadata(id).await?;
        let tags_str = if let Some(tags) = &new_metadata.tags {
            tags.join(",")
        } else {
            String::new()
        };

        let metadata_schema = self.metadata_table.schema().await?;
        let metadata_batch = RecordBatch::try_new(
            metadata_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id.to_string()])),
                Arc::new(BinaryArray::from(vec![new_metadata.file_path.as_bytes()])),
                Arc::new(StringArray::from(vec![
                    new_metadata.file_path_lossy.clone(),
                ])),
                Arc::new(StringArray::from(vec![new_metadata.file_type.as_str()])),
                Arc::new(StringArray::from(vec![new_metadata.mime_type.clone()])),
                Arc::new(UInt32Array::from(vec![
                    new_metadata.permissions.unwrap_or(0),
                ])), // nullable
                Arc::new(UInt64Array::from(vec![new_metadata.file_size])),
                Arc::new(Int64Array::from(vec![new_metadata.modified_time])),
                Arc::new(Int64Array::from(vec![
                    new_metadata.accessed_time.unwrap_or(0),
                ])), // nullable
                Arc::new(Int64Array::from(vec![
                    new_metadata.created_time.unwrap_or(0),
                ])), // nullable
                Arc::new(StringArray::from(vec![tags_str])),
            ],
        )?;

        let batches =
            RecordBatchIterator::new(vec![Ok(metadata_batch)].into_iter(), metadata_schema);
        self.metadata_table.add(Box::new(batches)).execute().await?;

        Ok(())
    }

    async fn delete_metadata(&self, id: &str) -> Result<()> {
        // Only delete metadata table entry, preserve all embeddings
        // This is called during metadata updates
        self.metadata_table
            .delete(&format!("id = '{}'", id))
            .await?;
        Ok(())
    }

    async fn delete_file_by_metadata_id(&self, id: &str) -> Result<()> {
        // Delete metadata AND all associated embeddings
        // This is for complete file deletion by metadata ID
        self.metadata_table
            .delete(&format!("id = '{}'", id))
            .await?;
        self.filename_embeddings_table
            .delete(&format!("metadata_id = '{}'", id))
            .await?;
        self.text_content_embeddings_table
            .delete(&format!("metadata_id = '{}'", id))
            .await?;
        self.image_content_embeddings_table
            .delete(&format!("metadata_id = '{}'", id))
            .await?;
        Ok(())
    }
}

impl LanceStore {
    pub async fn add_file(&self, params: &AddFile, content_table: &Table) -> Result<String> {
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().timestamp();

        if params.file_type != FileType::File {
            if params.content_embedding.is_some() || params.content_hash.is_some() {
                return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                    format!("Cannot store content for {:?}", params.file_type),
                )));
            }
        }

        if params.content_embedding.is_some() && params.content_hash.is_none() {
            return Err(lancedb::Error::from(ArrowError::InvalidArgumentError(
                "content_hash is required when storing content embeddings".to_string(),
            )));
        }

        let content_hash = params.content_hash.clone().unwrap_or_default();

        let tags_str = if let Some(tags) = &params.tags {
            tags.join(",")
        } else {
            String::new()
        };

        let metadata_schema = self.metadata_table.schema().await?;
        let metadata_batch = RecordBatch::try_new(
            metadata_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id.clone()])),
                Arc::new(BinaryArray::from(vec![params.file_path.as_bytes()])),
                Arc::new(StringArray::from(vec![params.file_path_lossy.clone()])),
                Arc::new(StringArray::from(vec![params.file_type.as_str()])),
                Arc::new(StringArray::from(vec![params.mime_type.clone()])),
                Arc::new(UInt32Array::from(vec![params.permissions])), // nullable
                Arc::new(UInt64Array::from(vec![params.file_size])),
                Arc::new(Int64Array::from(vec![params.modified_time])),
                Arc::new(Int64Array::from(vec![params.accessed_time])), // nullable
                Arc::new(Int64Array::from(vec![params.created_time])),  // nullable
                Arc::new(StringArray::from(vec![tags_str])),
            ],
        )?;

        let batches =
            RecordBatchIterator::new(vec![Ok(metadata_batch)].into_iter(), metadata_schema);
        self.metadata_table.add(Box::new(batches)).execute().await?;

        let filename_schema = self.filename_embeddings_table.schema().await?;
        let mut filename_builder = FixedSizeListBuilder::new(
            Float32Builder::new(),
            params.filename_embedding.len() as i32,
        );
        filename_builder
            .values()
            .append_slice(&params.filename_embedding);
        filename_builder.append(true);
        let filename_array = filename_builder.finish();

        let filename_batch = RecordBatch::try_new(
            filename_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![id.clone()])),
                Arc::new(filename_array),
                Arc::new(Int64Array::from(vec![now])),
            ],
        )?;

        let batches =
            RecordBatchIterator::new(vec![Ok(filename_batch)].into_iter(), filename_schema);
        self.filename_embeddings_table
            .add(Box::new(batches))
            .execute()
            .await?;

        if let Some(content_emb) = &params.content_embedding {
            let content_schema = content_table.schema().await?;
            let mut content_builder =
                FixedSizeListBuilder::new(Float32Builder::new(), content_emb.len() as i32);
            content_builder.values().append_slice(content_emb);
            content_builder.append(true);
            let content_array = content_builder.finish();

            let content_emb_batch = RecordBatch::try_new(
                content_schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![id.clone()])),
                    Arc::new(content_array),
                    Arc::new(StringArray::from(vec![content_hash])),
                    Arc::new(Int64Array::from(vec![now])),
                ],
            )?;

            let batches =
                RecordBatchIterator::new(vec![Ok(content_emb_batch)].into_iter(), content_schema);
            content_table.add(Box::new(batches)).execute().await?;
        }

        Ok(id)
    }

    pub async fn create_indices(&self) -> Result<()> {
        // obviously we need to create indices for the embeddings
        self.filename_embeddings_table
            .create_index(&["filename_embedding"], Index::Auto)
            .execute()
            .await?;
        self.text_content_embeddings_table
            .create_index(&["content_embedding"], Index::Auto)
            .execute()
            .await?;
        self.image_content_embeddings_table
            .create_index(&["content_embedding"], Index::Auto)
            .execute()
            .await?;

        // these are commonly searched fields
        self.metadata_table
            .create_index(&["mime_type"], Index::Auto)
            .execute()
            .await?;
        self.metadata_table
            .create_index(&["modified_time"], Index::Auto)
            .execute()
            .await?;
        self.metadata_table
            .create_index(&["file_path"], Index::Auto)
            .execute()
            .await?;

        // Maybe add tags?

        Ok(())
    }

    pub async fn cleanup_missing_files(&self) -> Result<u64> {
        // user triggered
        let mut deleted_count = 0;

        let query = self.metadata_table.query().limit(1000000);
        let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;

        let mut ids_to_delete = Vec::new();

        for batch in &batches {
            for row in 0..batch.num_rows() {
                match Self::extract_metadata_from_batch(batch, row) {
                    Ok(metadata) => {
                        let os_string = FilePath::decode_to_os_string(&metadata.file_path.bytes);
                        if !PathBuf::from(os_string).exists() {
                            ids_to_delete.push(metadata.id.clone());
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to extract metadata at row {}: {}", row, e);
                        continue;
                    }
                }
            }
        }

        for id in &ids_to_delete {
            self.metadata_table
                .delete(&format!("id = '{}'", id))
                .await?;
            self.filename_embeddings_table
                .delete(&format!("metadata_id = '{}'", id))
                .await?;
            self.text_content_embeddings_table
                .delete(&format!("metadata_id = '{}'", id))
                .await?;
            self.image_content_embeddings_table
                .delete(&format!("metadata_id = '{}'", id))
                .await?;

            deleted_count += 1;
        }

        Ok(deleted_count)
    }
}
