use quipu::{AddFile, FileType, LanceStore, create_mock_embedding};
use std::path::PathBuf;
use tempfile::tempdir;

pub async fn create_test_store() -> (LanceStore, tempfile::TempDir) {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_db");
    let store = LanceStore::new(db_path.to_str().unwrap()).await.unwrap();
    (store, temp_dir)
}

pub fn create_test_document_params() -> AddFile {
    AddFile::new(PathBuf::from("/path/to/test_document.txt"), FileType::File, create_mock_embedding("test_document.txt", "filename"))
        .mime_type("text/plain")
        .tags(&["test", "document"])
        .file_size(1024)
        .permissions(0o644)
}

pub fn create_test_directory_params() -> AddFile {
    AddFile::new(PathBuf::from("/path/to/test_dir"), FileType::Directory, create_mock_embedding("test_dir", "filename"))
        .tags(&["test", "directory"])
        .permissions(0o755)
}

pub fn create_test_symlink_params() -> AddFile {
    AddFile::new(PathBuf::from("/path/to/test_link"), FileType::Symlink, create_mock_embedding("test_link", "filename"))
        .tags(&["test", "symlink"])
        .permissions(0o777)
}

pub fn create_test_image_params() -> AddFile {
    let png_header = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let ihdr_chunk = vec![
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x03,
        0x00, 0x08, 0x02, 0x00, 0x00, 0x00,
    ];
    let content = [png_header, ihdr_chunk].concat();

    AddFile::new(PathBuf::from("/path/to/test_image.png"), FileType::File, create_mock_embedding("test_image.png", "filename"))
        .content_embedding(create_mock_embedding(&String::from_utf8_lossy(&content), "content"))
        .content_hash(&String::from_utf8_lossy(&content))
        .mime_type("image/png")
        .tags(&["image", "png"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

pub fn create_test_jpg_params() -> AddFile {
    let jpeg_header = vec![0xFF, 0xD8, 0xFF];
    let jfif_chunk = vec![
        0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01,
        0x00, 0x00,
    ];
    let content = [jpeg_header, jfif_chunk].concat();

    AddFile::new(PathBuf::from("/path/to/city_skyline.jpg"), FileType::File, create_mock_embedding("city_skyline.jpg", "filename"))
        .content_embedding(create_mock_embedding(&String::from_utf8_lossy(&content), "content"))
        .content_hash(&String::from_utf8_lossy(&content))
        .mime_type("image/jpeg")
        .tags(&["image", "jpg", "city", "skyline"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

// AI/ML related test parameters
pub fn create_ai_research_params() -> AddFile {
    let content = "Research paper on neural network architecture and machine learning performance benchmarks with extensive analysis of computational complexity and optimization strategies for deep learning models in production environments.";
    AddFile::new(PathBuf::from("/docs/ai_research.pdf"), FileType::File, create_mock_embedding("ai_research.pdf", "filename"))
        .content_embedding(create_mock_embedding(content, "content"))
        .content_hash(content)
        .mime_type("application/pdf")
        .tags(&["ai", "research"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

pub fn create_machine_learning_params() -> AddFile {
    let content = "Machine learning algorithms for pattern recognition including supervised, unsupervised, and reinforcement learning.";
    AddFile::new(PathBuf::from("/docs/machine_learning.txt"), FileType::File, create_mock_embedding("machine_learning.txt", "filename"))
        .content_embedding(create_mock_embedding(content, "content"))
        .content_hash(content)
        .mime_type("text/plain")
        .tags(&["machine_learning", "ai"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

pub fn create_cooking_recipe_params() -> AddFile {
    let content = "# Pasta Carbonara Recipe\n\n## Ingredients\n- 400g spaghetti\n- 200g pancetta or guanciale\n- 4 large eggs\n- 100g Pecorino Romano cheese\n- Black pepper\n- Salt\n\n## Instructions\n1. Cook pasta in salted water\n2. Fry pancetta until crispy\n3. Mix eggs and cheese\n4. Combine all ingredients\n5. Serve immediately with black pepper.";
    AddFile::new(PathBuf::from("/docs/cooking_recipe.md"), FileType::File, create_mock_embedding("cooking_recipe.md", "filename"))
        .content_embedding(create_mock_embedding(content, "content"))
        .content_hash(content)
        .mime_type("text/markdown")
        .tags(&["cooking", "recipe"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

pub fn create_tech_doc_params() -> AddFile {
    let content = "Technology overview covering neural networks, deep learning, computer vision and natural language processing.";
    AddFile::new(PathBuf::from("/docs/tech_doc.txt"), FileType::File, create_mock_embedding("tech_doc.txt", "filename"))
        .content_embedding(create_mock_embedding(content, "content"))
        .content_hash(content)
        .mime_type("text/plain")
        .tags(&["ai", "ml", "tech"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

pub fn create_recipe_params() -> AddFile {
    let content = "Cooking and baking recipe content: Learn to make delicious chocolate chip cookies, fluffy pancakes, and creamy pasta carbonara. Each recipe includes step-by-step instructions, ingredient lists, and cooking tips for perfect results every time.";
    AddFile::new(PathBuf::from("/docs/recipe.txt"), FileType::File, create_mock_embedding("recipe.txt", "filename"))
        .content_embedding(create_mock_embedding(content, "content"))
        .content_hash(content)
        .mime_type("text/plain")
        .tags(&["cooking", "recipe", "baking"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

pub fn create_ai_paper_params() -> AddFile {
    let content = "Research on transformer architecture optimization, attention mechanisms and training efficiency improvements.";
    AddFile::new(PathBuf::from("/docs/ai_paper.pdf"), FileType::File, create_mock_embedding("ai_paper.pdf", "filename"))
        .content_embedding(create_mock_embedding(content, "content"))
        .content_hash(content)
        .mime_type("application/pdf")
        .tags(&["ai", "research", "paper"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}

pub fn create_recipe_book_params() -> AddFile {
    let content = "Comprehensive cooking guide with Italian pasta, French pastries, Asian stir-fries and American comfort food recipes.";
    AddFile::new(PathBuf::from("/docs/recipe_book.txt"), FileType::File, create_mock_embedding("recipe_book.txt", "filename"))
        .content_embedding(create_mock_embedding(content, "content"))
        .content_hash(content)
        .mime_type("text/plain")
        .tags(&["cooking", "recipes", "book"])
        .file_size(content.len() as u64)
        .permissions(0o644)
}
