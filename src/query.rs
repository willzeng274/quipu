use arrow_array::{RecordBatch, StringArray};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::Result;
use std::collections::HashMap;

use crate::store::{FileMetadata, LanceStore};

#[derive(Debug, Clone)]
pub enum VectorType {
    FileName,
    TextContent,
    ImageContent,
    Combined { filename_weight: f32, text_weight: f32, image_weight: f32 },
}

#[derive(Debug, Clone)]
struct MetadataFilter {
    conditions: Vec<String>,
}

impl MetadataFilter {
    fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    fn add_condition(&mut self, condition: String) {
        self.conditions.push(condition);
    }

    fn build(&self) -> Option<String> {
        if self.conditions.is_empty() {
            None
        } else {
            Some(self.conditions.join(" AND "))
        }
    }
}

pub struct QueryBuilder {
    store: LanceStore,
    metadata_filter: MetadataFilter,
    embeddings: Option<Vec<f32>>,
    vector_type: Option<VectorType>,
    limit: Option<usize>,
}

impl QueryBuilder {
    pub fn new(store: LanceStore) -> Self {
        Self {
            store,
            metadata_filter: MetadataFilter::new(),
            embeddings: None,
            vector_type: None,
            limit: None,
        }
    }

    pub fn semantic(mut self, embeddings: &[f32], vector_type: VectorType) -> Self {
        self.embeddings = Some(embeddings.to_vec());
        self.vector_type = Some(vector_type);
        self
    }

    pub fn by_filename(mut self, embeddings: &[f32]) -> Self {
        self.embeddings = Some(embeddings.to_vec());
        self.vector_type = Some(VectorType::FileName);
        self
    }

    pub fn by_text_content(mut self, embeddings: &[f32]) -> Self {
        self.embeddings = Some(embeddings.to_vec());
        self.vector_type = Some(VectorType::TextContent);
        self
    }

    pub fn by_image_content(mut self, embeddings: &[f32]) -> Self {
        self.embeddings = Some(embeddings.to_vec());
        self.vector_type = Some(VectorType::ImageContent);
        self
    }

    pub fn combined(mut self, embeddings: &[f32]) -> Self {
        self.embeddings = Some(embeddings.to_vec());
        self.vector_type = Some(VectorType::Combined {
            filename_weight: 0.3,
            text_weight: 0.4,
            image_weight: 0.3,
        });
        self
    }

    pub fn combined_with_weights(mut self, embeddings: &[f32], filename_weight: f32, text_weight: f32, image_weight: f32) -> Self {
        self.embeddings = Some(embeddings.to_vec());
        self.vector_type = Some(VectorType::Combined {
            filename_weight,
            text_weight,
            image_weight,
        });
        self
    }

    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn filter(mut self, condition: &str) -> Self {
        self.metadata_filter.add_condition(condition.to_string());
        self
    }

    pub fn mime_type(mut self, mime_type: &str) -> Self {
        self.metadata_filter
            .add_condition(format!("mime_type = '{}'", mime_type));
        self
    }

    pub fn file_type(mut self, file_type: &str) -> Self {
        self.metadata_filter
            .add_condition(format!("file_type = '{}'", file_type));
        self
    }

    pub fn path_contains(mut self, path_segment: &str) -> Self {
        self.metadata_filter
            .add_condition(format!("file_path_lossy LIKE '%{}%'", path_segment));
        self
    }

    pub fn path_starts_with(mut self, prefix: &str) -> Self {
        self.metadata_filter
            .add_condition(format!("file_path_lossy LIKE '{}%'", prefix));
        self
    }

    pub fn path_ends_with(mut self, suffix: &str) -> Self {
        self.metadata_filter
            .add_condition(format!("file_path_lossy LIKE '%{}'", suffix));
        self
    }

    pub fn extension(mut self, ext: &str) -> Self {
        let ext_with_dot = if ext.starts_with('.') {
            ext.to_string()
        } else {
            format!(".{}", ext)
        };
        self.metadata_filter
            .add_condition(format!("file_path_lossy LIKE '%{}'", ext_with_dot));
        self
    }

    pub fn size_range(mut self, min: u64, max: u64) -> Self {
        self.metadata_filter
            .add_condition(format!("file_size >= {} AND file_size <= {}", min, max));
        self
    }

    pub fn size_min(mut self, min: u64) -> Self {
        self.metadata_filter
            .add_condition(format!("file_size >= {}", min));
        self
    }

    pub fn size_max(mut self, max: u64) -> Self {
        self.metadata_filter
            .add_condition(format!("file_size <= {}", max));
        self
    }

    pub fn modified_after(mut self, timestamp: i64) -> Self {
        self.metadata_filter
            .add_condition(format!("modified_time > {}", timestamp));
        self
    }

    pub fn modified_before(mut self, timestamp: i64) -> Self {
        self.metadata_filter
            .add_condition(format!("modified_time < {}", timestamp));
        self
    }

    pub fn modified_between(mut self, start: i64, end: i64) -> Self {
        self.metadata_filter
            .add_condition(format!("modified_time >= {} AND modified_time <= {}", start, end));
        self
    }

    pub fn created_after(mut self, timestamp: i64) -> Self {
        self.metadata_filter
            .add_condition(format!("created_time > {}", timestamp));
        self
    }

    pub fn created_before(mut self, timestamp: i64) -> Self {
        self.metadata_filter
            .add_condition(format!("created_time < {}", timestamp));
        self
    }

    pub fn accessed_after(mut self, timestamp: i64) -> Self {
        self.metadata_filter
            .add_condition(format!("accessed_time > {}", timestamp));
        self
    }

    pub fn has_any_tags(mut self, tags: &[&str]) -> Self {
        if !tags.is_empty() {
            let tag_conditions: Vec<String> = tags
                .iter()
                .map(|tag| format!("tags LIKE '%{}%'", tag))
                .collect();
            self.metadata_filter
                .add_condition(format!("({})", tag_conditions.join(" OR ")));
        }
        self
    }

    pub fn has_all_tags(mut self, tags: &[&str]) -> Self {
        for tag in tags {
            self.metadata_filter
                .add_condition(format!("tags LIKE '%{}%'", tag));
        }
        self
    }

    pub fn has_tag(mut self, tag: &str) -> Self {
        self.metadata_filter
            .add_condition(format!("tags LIKE '%{}%'", tag));
        self
    }

    pub fn has_permissions(mut self, permissions: u32) -> Self {
        self.metadata_filter
            .add_condition(format!("permissions = {}", permissions));
        self
    }

    // NOTE: binary & DOES NOT WORK in LanceDB
    pub fn is_readable(mut self) -> Self {
        // Check for common readable permissions
        self.metadata_filter
            .add_condition("permissions IN (644, 755, 777, 600, 400, 444, 664, 666)".to_string());
        self
    }

    pub fn is_writable(mut self) -> Self {
        // Check for common writable permissions
        self.metadata_filter
            .add_condition("permissions IN (644, 755, 777, 600, 200, 664, 666, 622, 722)".to_string());
        self
    }

    pub fn is_executable(mut self) -> Self {
        // Check for common executable permissions
        self.metadata_filter
            .add_condition("permissions IN (755, 777, 700, 100, 711, 733, 555, 111)".to_string());
        self
    }

    pub async fn execute(self) -> Result<Vec<FileMetadata>> {
        match (&self.embeddings, &self.vector_type, self.metadata_filter.build()) {
            (None, None, None) => self.execute_scan().await,
            (None, None, Some(filter)) => self.execute_metadata_only(filter).await,
            (Some(embeddings), Some(vector_type), None) => {
                self.execute_semantic_only(embeddings, vector_type).await
            }
            (Some(embeddings), Some(vector_type), Some(filter)) => {
                self.execute_semantic_with_filter(embeddings, vector_type, filter).await
            }
            _ => Err(lancedb::Error::from(arrow_schema::ArrowError::InvalidArgumentError(
                "Invalid query configuration: embeddings and vector_type must be provided together".to_string(),
            ))),
        }
    }

    async fn execute_scan(&self) -> Result<Vec<FileMetadata>> {
        let limit = self.limit.unwrap_or(1_000_000);
        let query = self.store.metadata_table.query().limit(limit);
        
        let mut results = Vec::new();
        let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
        
        for batch in &batches {
            for row in 0..batch.num_rows() {
                match LanceStore::extract_metadata_from_batch(batch, row) {
                    Ok(metadata) => results.push(metadata),
                    Err(e) => {
                        eprintln!("Failed to extract metadata at row {}: {}", row, e);
                    }
                }
            }
        }
        
        Ok(results)
    }

    async fn execute_metadata_only(&self, filter: String) -> Result<Vec<FileMetadata>> {
        let limit = self.limit.unwrap_or(1_000_000);
        let query = self.store.metadata_table.query()
            .only_if(filter)
            .limit(limit);
        
        let mut results = Vec::new();
        let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
        
        for batch in &batches {
            for row in 0..batch.num_rows() {
                match LanceStore::extract_metadata_from_batch(batch, row) {
                    Ok(metadata) => results.push(metadata),
                    Err(e) => {
                        eprintln!("Failed to extract metadata at row {}: {}", row, e);
                    }
                }
            }
        }
        
        Ok(results)
    }

    async fn execute_semantic_only(&self, embeddings: &[f32], vector_type: &VectorType) -> Result<Vec<FileMetadata>> {
        let limit = self.limit.unwrap_or(10);
        
        match vector_type {
            VectorType::FileName => {
                self.search_by_filename(embeddings, limit).await
            }
            VectorType::TextContent => {
                self.search_by_content(
                    embeddings,
                    limit,
                    &self.store.text_content_embeddings_table,
                ).await
            }
            VectorType::ImageContent => {
                self.search_by_content(
                    embeddings,
                    limit,
                    &self.store.image_content_embeddings_table,
                ).await
            }
            VectorType::Combined { filename_weight, text_weight, image_weight } => {
                self.execute_combined_search(embeddings, *filename_weight, *text_weight, *image_weight, None).await
            }
        }
    }

    async fn execute_semantic_with_filter(
        &self,
        embeddings: &[f32],
        vector_type: &VectorType,
        filter: String,
    ) -> Result<Vec<FileMetadata>> {
        let limit = self.limit.unwrap_or(10);
        
        match vector_type {
            VectorType::FileName => {
                let metadata_ids = self.get_filtered_metadata_ids(&filter).await?;
                if metadata_ids.is_empty() {
                    return Ok(Vec::new());
                }
                let id_filter = self.build_id_filter(&metadata_ids);
                self.search_by_filename_filtered(embeddings, limit, &id_filter).await
            }
            VectorType::TextContent => {
                let metadata_ids = self.get_filtered_metadata_ids(&filter).await?;
                if metadata_ids.is_empty() {
                    return Ok(Vec::new());
                }
                let id_filter = self.build_id_filter(&metadata_ids);
                self.search_by_content_filtered(
                    embeddings,
                    limit,
                    &id_filter,
                    &self.store.text_content_embeddings_table,
                ).await
            }
            VectorType::ImageContent => {
                let metadata_ids = self.get_filtered_metadata_ids(&filter).await?;
                if metadata_ids.is_empty() {
                    return Ok(Vec::new());
                }
                let id_filter = self.build_id_filter(&metadata_ids);
                self.search_by_content_filtered(
                    embeddings,
                    limit,
                    &id_filter,
                    &self.store.image_content_embeddings_table,
                ).await
            }
            VectorType::Combined { filename_weight, text_weight, image_weight } => {
                self.execute_combined_search(embeddings, *filename_weight, *text_weight, *image_weight, Some(filter)).await
            }
        }
    }

    fn build_id_filter(&self, metadata_ids: &[String]) -> String {
        format!(
            "metadata_id IN ({})",
            metadata_ids
                .iter()
                .map(|id| format!("'{}'", id))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    async fn execute_combined_search(
        &self,
        embeddings: &[f32],
        filename_weight: f32,
        text_weight: f32,
        image_weight: f32,
        metadata_filter: Option<String>,
    ) -> Result<Vec<FileMetadata>> {
        let limit = self.limit.unwrap_or(10);
        let search_limit = limit * 3;

        let id_filter = if let Some(filter) = metadata_filter {
            let metadata_ids = self.get_filtered_metadata_ids(&filter).await?;
            if metadata_ids.is_empty() {
                return Ok(Vec::new());
            }
            Some(self.build_id_filter(&metadata_ids))
        } else {
            None
        };

        let (filename_results, text_results, image_results) = tokio::join!(
            self.search_with_optional_filter(embeddings, search_limit, &self.store.filename_embeddings_table, &id_filter),
            self.search_with_optional_filter(embeddings, search_limit, &self.store.text_content_embeddings_table, &id_filter),
            self.search_with_optional_filter(embeddings, search_limit, &self.store.image_content_embeddings_table, &id_filter)
        );

        let mut file_scores: HashMap<String, (FileMetadata, f32)> = HashMap::new();

        if let Ok(results) = filename_results {
            for (i, metadata) in results.into_iter().enumerate() {
                let score = filename_weight * (1.0 - (i as f32 / search_limit as f32));
                file_scores.insert(metadata.id.clone(), (metadata, score));
            }
        }

        if let Ok(results) = text_results {
            for (i, metadata) in results.into_iter().enumerate() {
                let score = text_weight * (1.0 - (i as f32 / search_limit as f32));
                file_scores
                    .entry(metadata.id.clone())
                    .and_modify(|(_, s)| *s += score)
                    .or_insert((metadata, score));
            }
        }

        if let Ok(results) = image_results {
            for (i, metadata) in results.into_iter().enumerate() {
                let score = image_weight * (1.0 - (i as f32 / search_limit as f32));
                file_scores
                    .entry(metadata.id.clone())
                    .and_modify(|(_, s)| *s += score)
                    .or_insert((metadata, score));
            }
        }

        let mut results: Vec<_> = file_scores.into_values().collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(limit);

        Ok(results.into_iter().map(|(metadata, _)| metadata).collect())
    }

    async fn search_with_optional_filter(
        &self,
        embeddings: &[f32],
        limit: usize,
        table: &lancedb::Table,
        filter: &Option<String>,
    ) -> Result<Vec<FileMetadata>> {
        let query = table.vector_search(embeddings)?.limit(limit);
        let query = if let Some(filter) = filter {
            query.only_if(filter)
        } else {
            query
        };
        self.collect_metadata_from_embeddings(query).await
    }

    async fn get_filtered_metadata_ids(&self, filter: &str) -> Result<Vec<String>> {
        let query = self.store.metadata_table.query()
            .only_if(filter)
            .limit(1_000_000);
        
        let mut ids = Vec::new();
        let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
        
        for batch in &batches {
            if let Some(id_col) = batch.column_by_name("id") {
                if let Some(id_array) = id_col.as_any().downcast_ref::<StringArray>() {
                    for i in 0..batch.num_rows() {
                        ids.push(id_array.value(i).to_string());
                    }
                }
            }
        }
        
        Ok(ids)
    }

    async fn search_by_filename(
        &self,
        embeddings: &[f32],
        limit: usize,
    ) -> Result<Vec<FileMetadata>> {
        let query = self.store.filename_embeddings_table
            .vector_search(embeddings)?
            .limit(limit);
        
        self.collect_metadata_from_embeddings(query).await
    }

    async fn search_by_filename_filtered(
        &self,
        embeddings: &[f32],
        limit: usize,
        filter: &str,
    ) -> Result<Vec<FileMetadata>> {
        let query = self.store.filename_embeddings_table
            .vector_search(embeddings)?
            .only_if(filter)
            .limit(limit);
        
        self.collect_metadata_from_embeddings(query).await
    }

    async fn search_by_content(
        &self,
        embeddings: &[f32],
        limit: usize,
        table: &lancedb::Table,
    ) -> Result<Vec<FileMetadata>> {
        let query = table.vector_search(embeddings)?.limit(limit);
        self.collect_metadata_from_embeddings(query).await
    }

    async fn search_by_content_filtered(
        &self,
        embeddings: &[f32],
        limit: usize,
        filter: &str,
        table: &lancedb::Table,
    ) -> Result<Vec<FileMetadata>> {
        let query = table
            .vector_search(embeddings)?
            .only_if(filter)
            .limit(limit);
        
        self.collect_metadata_from_embeddings(query).await
    }

    async fn collect_metadata_from_embeddings<Q>(
        &self,
        query: Q,
    ) -> Result<Vec<FileMetadata>>
    where
        Q: QueryBase + ExecutableQuery + Send,
    {
        let mut results = Vec::new();
        let batches: Vec<RecordBatch> = query.execute().await?.try_collect().await?;
        
        for batch in &batches {
            if let Some(metadata_id_col) = batch.column_by_name("metadata_id") {
                if let Some(metadata_ids) = metadata_id_col
                    .as_any()
                    .downcast_ref::<StringArray>()
                {
                    for i in 0..batch.num_rows() {
                        let metadata_id = metadata_ids.value(i);
                        if let Some(metadata) = self.store.get_metadata(metadata_id).await? {
                            results.push(metadata);
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
}

impl LanceStore {
    pub fn query(&self) -> QueryBuilder {
        QueryBuilder::new(self.clone())
    }
}
