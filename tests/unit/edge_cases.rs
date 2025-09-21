use quipu::{FileType, create_mock_embedding, compute_md5_hash};

#[test]
fn test_create_mock_embedding_consistency() {
    let embedding1 = create_mock_embedding("test content", "content");
    let embedding2 = create_mock_embedding("test content", "content");
    let embedding3 = create_mock_embedding("different content", "content");

    assert_eq!(embedding1, embedding2);

    assert_ne!(embedding1, embedding3);

    assert_eq!(embedding1.len(), 384);
    assert_eq!(embedding2.len(), 384);
    assert_eq!(embedding3.len(), 384);
}

#[test]
fn test_create_mock_embedding_different_types() {
    let filename_embedding = create_mock_embedding("file.txt", "filename");
    let content_embedding = create_mock_embedding("file content", "content");

    assert_ne!(filename_embedding, content_embedding);

    assert_eq!(filename_embedding.len(), 384);
    assert_eq!(content_embedding.len(), 384);
}

#[test]
fn test_compute_md5_hash_consistency() {
    let hash1 = compute_md5_hash("test content");
    let hash2 = compute_md5_hash("test content");
    let hash3 = compute_md5_hash("different content");

    assert_eq!(hash1, hash2);

    assert_ne!(hash1, hash3);

    assert!(!hash1.is_empty());
    assert!(!hash2.is_empty());
    assert!(!hash3.is_empty());
}

#[test]
fn test_compute_md5_hash_empty_string() {
    let empty_hash = compute_md5_hash("");
    let content_hash = compute_md5_hash("content");

    assert!(!empty_hash.is_empty());

    assert_ne!(empty_hash, content_hash);
}

#[test]
fn test_file_type_as_str() {
    assert_eq!(FileType::File.as_str(), "file");
    assert_eq!(FileType::Directory.as_str(), "directory");
    assert_eq!(FileType::Symlink.as_str(), "symlink");
}

#[test]
fn test_file_type_from_str() {
    assert_eq!(FileType::from_str("file"), Some(FileType::File));
    assert_eq!(FileType::from_str("directory"), Some(FileType::Directory));
    assert_eq!(FileType::from_str("symlink"), Some(FileType::Symlink));
    assert_eq!(FileType::from_str("unknown"), None);
    assert_eq!(FileType::from_str(""), None);
}

#[test]
fn test_file_type_from_str_case_sensitivity() {
    assert_eq!(FileType::from_str("FILE"), None);
    assert_eq!(FileType::from_str("File"), None);
    assert_eq!(FileType::from_str("DIRECTORY"), None);
    assert_eq!(FileType::from_str("Directory"), None);
}

#[test]
fn test_embedding_vector_properties() {
    let embedding = create_mock_embedding("test", "content");

    assert!(!embedding.is_empty());

    assert_eq!(embedding.len(), 384);

    for &value in &embedding {
        assert!(value.is_finite(), "Embedding contains non-finite value: {}", value);
    }

    let sum: f32 = embedding.iter().sum();
    assert!(sum != 0.0, "Embedding is all zeros");
}

#[test]
fn test_md5_hash_format() {
    let hash = compute_md5_hash("test content");

    assert_eq!(hash.len(), 32);

    for c in hash.chars() {
        assert!(c.is_ascii_hexdigit(), "Hash contains non-hex character: {}", c);
    }
}

#[test]
fn test_special_characters_in_content() {
    let special_content = "Special chars: àáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ";
    let embedding = create_mock_embedding(special_content, "content");
    let hash = compute_md5_hash(special_content);

    assert_eq!(embedding.len(), 384);
    assert_eq!(hash.len(), 32);

    for &value in &embedding {
        assert!(value.is_finite());
    }
}

#[test]
fn test_unicode_content() {
    let unicode_content = "Unicode: 🚀 🔥 💯 🌟 🎉 📁 📄 🔗";
    let embedding = create_mock_embedding(unicode_content, "content");
    let hash = compute_md5_hash(unicode_content);

    assert_eq!(embedding.len(), 384);
    assert_eq!(hash.len(), 32);


    for &value in &embedding {
        assert!(value.is_finite());
    }
}
