//! Storage Adapter for audiopipe-server
//!
//! Implements the `AudioPipeStorage` trait for persisting pipeline results.
//! Uses Pro-specific tables that don't modify the OSS schema.
//!
//! Note: This is a simplified adapter for audiopipe-server. License management
//! is deferred - can_write_pro_data() always returns true.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use sqlx::SqlitePool;

use audiopipe::storage::{
    AudioPipeStorage, AudioTranscription, ClassificationJob, DiarizationJob, MergeHistoryEntry,
    ReprocessQueueItem, Speaker, SpeakerBlockInput, SpeakerBlockOutput, SpeakerBlockPendingSummary,
    SpeakerEmbeddingRow, SpeakerEmbeddingV2, StorageError, StorageResult, StoredChunk,
    StoredSpeaker, StoredTopicGroup, TopicGroupInput, TranscriptionInput,
    UnifiedTranscriptJob, UnifiedTranscriptSegmentInput,
};
use audiopipe::PipelineOutput;

// Type alias for async trait methods
type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// Pro-specific migrations embedded at compile time.
/// These create new tables without modifying the OSS schema.
const MIGRATIONS: &[(&str, &str)] = &[
    (
        "001_embedding_extraction_events",
        include_str!("../../audiopipe/migrations/001_embedding_extraction_events.sql"),
    ),
    (
        "002_topic_groups",
        include_str!("../../audiopipe/migrations/002_topic_groups.sql"),
    ),
];

/// Storage adapter that bridges audiopipe to screenpipe's database
///
/// Uses Pro-specific tables. License checking is deferred.
pub struct ScreenpipeStorage {
    pool: Arc<SqlitePool>,
}

impl ScreenpipeStorage {
    /// Create a new storage adapter from a database path.
    /// Runs Pro-specific migrations to create tables if they don't exist.
    pub async fn new(db_path: &str) -> Result<Self, sqlx::Error> {
        let pool = SqlitePool::connect(&format!("sqlite:{}?mode=rwc", db_path)).await?;
        let storage = Self {
            pool: Arc::new(pool),
        };

        // Run migrations to ensure Pro tables exist
        storage.run_migrations().await?;

        Ok(storage)
    }

    /// Run Pro-specific migrations.
    /// All migrations use CREATE TABLE/INDEX IF NOT EXISTS, making them idempotent.
    async fn run_migrations(&self) -> Result<(), sqlx::Error> {
        for (name, sql) in MIGRATIONS {
            tracing::debug!("running migration: {}", name);

            // Execute each statement in the migration
            // Split by semicolon but handle empty statements
            for statement in sql.split(';') {
                let stmt = statement.trim();
                if stmt.is_empty() || stmt.starts_with("--") {
                    continue;
                }

                sqlx::query(stmt).execute(self.pool.as_ref()).await?;
            }

            tracing::info!("migration complete: {}", name);
        }

        Ok(())
    }

    /// Create from an existing pool
    pub fn from_pool(pool: Arc<SqlitePool>) -> Self {
        Self { pool }
    }

    /// Get the underlying pool (for direct queries if needed)
    pub fn pool(&self) -> &SqlitePool {
        &self.pool
    }
}

// Helper to convert database errors to storage errors
fn db_err(e: impl std::fmt::Display) -> StorageError {
    StorageError::QueryError(e.to_string())
}

// Extract device name from file path
// Expected format: `{device_name}_{YYYY-MM-DD_HH-MM-SS}.{ext}`
fn extract_device_name_from_path(file_path: &str) -> String {
    use std::path::Path;

    let filename = Path::new(file_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    // Find the last underscore followed by a date pattern (YYYY-MM-DD)
    // The device name is everything before that
    if let Some(idx) = filename.rfind('_') {
        let after = &filename[idx + 1..];
        // Check if it looks like a date (starts with 4 digits)
        if after.len() >= 10 && after.chars().take(4).all(|c| c.is_ascii_digit()) {
            // Look for the previous underscore (before the date)
            let before_date = &filename[..idx];
            if let Some(date_start_idx) = before_date.rfind('_') {
                let potential_date_prefix = &before_date[date_start_idx + 1..];
                // If this also looks like date part, use everything before it
                if potential_date_prefix.chars().take(4).all(|c| c.is_ascii_digit()) {
                    return before_date[..date_start_idx].to_string();
                }
            }
            return before_date.to_string();
        }
    }

    filename.to_string()
}

// Helper to parse date string to DateTime<Utc>
fn parse_date(s: &str) -> DateTime<Utc> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now())
}

impl AudioPipeStorage for ScreenpipeStorage {
    // === Chunk Operations ===

    fn store_chunk(&self, output: &PipelineOutput) -> BoxFuture<'_, StorageResult<i64>> {
        let pool = self.pool.clone();
        let file_path = output.audio_file_path.to_string_lossy().to_string();
        Box::pin(async move {
            // First try to get existing chunk by file_path
            let existing = sqlx::query_scalar::<_, i64>(
                "SELECT id FROM audio_chunks WHERE file_path = ?1",
            )
            .bind(&file_path)
            .fetch_optional(pool.as_ref())
            .await
            .map_err(db_err)?;

            if let Some(id) = existing {
                return Ok(id);
            }

            // Insert new chunk
            let result = sqlx::query_scalar::<_, i64>(
                r#"
                INSERT INTO audio_chunks (file_path, timestamp)
                VALUES (?1, datetime('now'))
                RETURNING id
                "#,
            )
            .bind(&file_path)
            .fetch_one(pool.as_ref())
            .await
            .map_err(db_err)?;
            Ok(result)
        })
    }

    fn get_chunk(&self, id: i64) -> BoxFuture<'_, StorageResult<Option<StoredChunk>>> {
        let pool = self.pool.clone();
        Box::pin(async move {
            let row = sqlx::query_as::<_, (i64, String, Option<String>)>(
                r#"
                SELECT id, file_path, timestamp
                FROM audio_chunks
                WHERE id = ?1
                "#,
            )
            .bind(id)
            .fetch_optional(pool.as_ref())
            .await
            .map_err(db_err)?;

            match row {
                Some((id, file_path, timestamp)) => Ok(Some(StoredChunk {
                    id,
                    audio_file_path: file_path.clone(),
                    device_name: extract_device_name_from_path(&file_path),
                    device_type: "input".to_string(),
                    timestamp: timestamp.map(|t| parse_date(&t)).unwrap_or_else(Utc::now),
                    duration_ms: 30000,
                    quality_score: 0.0,
                    snr_db: 0.0,
                    volume_dbfs: -60.0,
                    preprocessing_applied: false,
                    speech_ratio: 0.0,
                    vad_segments: vec![],
                    primary_text: String::new(),
                    secondary_text: None,
                    confidence: 0.0,
                    outcome_type: "unknown".to_string(),
                    word_timestamps: vec![],
                    scene_classes: None,
                    created_at: Utc::now(),
                })),
                None => Ok(None),
            }
        })
    }

    fn get_chunks_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: usize,
    ) -> BoxFuture<'_, StorageResult<Vec<StoredChunk>>> {
        let pool = self.pool.clone();
        Box::pin(async move {
            let start_str = start.to_rfc3339();
            let end_str = end.to_rfc3339();

            let rows = sqlx::query_as::<_, (i64, String, Option<String>)>(
                r#"
                SELECT id, file_path, timestamp
                FROM audio_chunks
                WHERE timestamp >= ?1 AND timestamp <= ?2
                ORDER BY timestamp DESC
                LIMIT ?3
                "#,
            )
            .bind(&start_str)
            .bind(&end_str)
            .bind(limit as i32)
            .fetch_all(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(rows
                .into_iter()
                .map(|(id, file_path, timestamp)| StoredChunk {
                    id,
                    audio_file_path: file_path.clone(),
                    device_name: extract_device_name_from_path(&file_path),
                    device_type: "input".to_string(),
                    timestamp: timestamp.map(|t| parse_date(&t)).unwrap_or_else(Utc::now),
                    duration_ms: 30000,
                    quality_score: 0.0,
                    snr_db: 0.0,
                    volume_dbfs: -60.0,
                    preprocessing_applied: false,
                    speech_ratio: 0.0,
                    vad_segments: vec![],
                    primary_text: String::new(),
                    secondary_text: None,
                    confidence: 0.0,
                    outcome_type: "unknown".to_string(),
                    word_timestamps: vec![],
                    scene_classes: None,
                    created_at: Utc::now(),
                })
                .collect())
        })
    }

    // === Pro Pipeline Results ===

    fn store_pro_pipeline_result(
        &self,
        result: &audiopipe::ProPipelineResultInput,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        let pool = self.pool.clone();
        let audio_chunk_id = result.audio_chunk_id;
        let vad_speech_ratio = result.vad_speech_ratio.map(|v| v as f64);
        let vad_segment_count = result.vad_segment_count;
        let vad_passed = result.vad_passed;
        let quality_score = result.quality_score.map(|v| v as f64);
        let volume_dbfs = result.volume_dbfs.map(|v| v as f64);
        let snr_db = result.snr_db.map(|v| v as f64);
        let transcription_confidence = result.transcription_confidence.map(|v| v as f64);
        let transcription_word_count = result.transcription_word_count;
        let transcription_outcome = result.transcription_outcome.clone();
        let processing_duration_ms = result.processing_duration_ms.map(|v| v as i32);
        let pipeline_version = result.pipeline_version.clone();

        Box::pin(async move {
            let id = sqlx::query_scalar::<_, i64>(
                r#"
                INSERT INTO pro_pipeline_results (
                    audio_chunk_id, vad_speech_ratio, vad_segment_count, vad_passed,
                    quality_score, volume_dbfs, snr_db,
                    transcription_confidence, transcription_word_count, transcription_outcome,
                    processing_duration_ms, pipeline_version, created_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, datetime('now'))
                ON CONFLICT(audio_chunk_id) DO UPDATE SET
                    vad_speech_ratio = excluded.vad_speech_ratio,
                    vad_segment_count = excluded.vad_segment_count,
                    vad_passed = excluded.vad_passed,
                    quality_score = excluded.quality_score,
                    volume_dbfs = excluded.volume_dbfs,
                    snr_db = excluded.snr_db,
                    transcription_confidence = excluded.transcription_confidence,
                    transcription_word_count = excluded.transcription_word_count,
                    transcription_outcome = excluded.transcription_outcome,
                    processing_duration_ms = excluded.processing_duration_ms,
                    pipeline_version = excluded.pipeline_version
                RETURNING id
                "#,
            )
            .bind(audio_chunk_id)
            .bind(vad_speech_ratio)
            .bind(vad_segment_count)
            .bind(vad_passed)
            .bind(quality_score)
            .bind(volume_dbfs)
            .bind(snr_db)
            .bind(transcription_confidence)
            .bind(transcription_word_count)
            .bind(&transcription_outcome)
            .bind(processing_duration_ms)
            .bind(&pipeline_version)
            .fetch_one(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(id)
        })
    }

    fn get_pro_pipeline_result(
        &self,
        audio_chunk_id: i64,
    ) -> BoxFuture<'_, StorageResult<Option<audiopipe::ProPipelineResultInput>>> {
        let pool = self.pool.clone();
        Box::pin(async move {
            // Query Pro pipeline result
            let row = sqlx::query_as::<_, (
                i64, Option<f64>, Option<i32>, Option<bool>,
                Option<f64>, Option<f64>, Option<f64>,
                Option<f64>, Option<i32>, Option<String>,
                Option<i32>, String,
            )>(
                r#"
                SELECT
                    audio_chunk_id, vad_speech_ratio, vad_segment_count, vad_passed,
                    quality_score, volume_dbfs, snr_db,
                    transcription_confidence, transcription_word_count, transcription_outcome,
                    processing_duration_ms, pipeline_version
                FROM pro_pipeline_results
                WHERE audio_chunk_id = ?1
                "#,
            )
            .bind(audio_chunk_id)
            .fetch_optional(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(row.map(|r| audiopipe::ProPipelineResultInput {
                audio_chunk_id: r.0,
                vad_speech_ratio: r.1.map(|v| v as f32),
                vad_segment_count: r.2,
                vad_passed: r.3,
                quality_score: r.4.map(|v| v as f32),
                volume_dbfs: r.5.map(|v| v as f32),
                volume_max_dbfs: None,
                snr_db: r.6.map(|v| v as f32),
                audio_level_passed: None,
                hallucination_score: None,
                hallucination_patterns: None,
                hallucination_passed: None,
                transcription_confidence: r.7.map(|v| v as f32),
                transcription_word_count: r.8,
                transcription_outcome: r.9,
                filter_reason: None,
                scene_top_class: None,
                scene_scores: None,
                emotion_label: None,
                emotion_confidence: None,
                language_code: None,
                language_confidence: None,
                speaker_count: None,
                processing_duration_ms: r.10.map(|v| v as i64),
                pipeline_version: r.11,
                conversation_id: None,
            }))
        })
    }

    fn can_write_pro_data(&self) -> BoxFuture<'_, bool> {
        // License management deferred - always allow writes
        Box::pin(async { true })
    }

    // === Stub implementations for remaining trait methods ===
    // These will be implemented as needed in future stories

    fn queue_for_reprocessing(
        &self,
        _chunk_id: i64,
        _reason: &str,
        _confidence: f32,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn get_reprocess_queue(
        &self,
        _limit: usize,
    ) -> BoxFuture<'_, StorageResult<Vec<ReprocessQueueItem>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn complete_reprocess_item(&self, _id: i64, _status: &str) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn claim_classification_job(&self) -> BoxFuture<'_, StorageResult<Option<ClassificationJob>>> {
        Box::pin(async { Ok(None) })
    }

    fn complete_classification_job(&self, _job_id: i64) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn fail_classification_job(
        &self,
        _job_id: i64,
        _error_msg: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_audio_chunk_path(
        &self,
        audio_chunk_id: i64,
    ) -> BoxFuture<'_, StorageResult<Option<String>>> {
        let pool = self.pool.clone();
        Box::pin(async move {
            let path = sqlx::query_scalar::<_, String>(
                "SELECT file_path FROM audio_chunks WHERE id = ?1",
            )
            .bind(audio_chunk_id)
            .fetch_optional(pool.as_ref())
            .await
            .map_err(db_err)?;
            Ok(path)
        })
    }

    fn has_audio_classification(&self, _audio_chunk_id: i64) -> BoxFuture<'_, StorageResult<bool>> {
        Box::pin(async { Ok(false) })
    }

    fn queue_classification_job(
        &self,
        _audio_chunk_id: i64,
        _priority: i32,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn insert_audio_classification(
        &self,
        _audio_chunk_id: i64,
        _inaspeech_json: &str,
        _yamnet_json: &str,
        _processing_time_ms: Option<i64>,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn update_transcription_classification_metadata(
        &self,
        _audio_chunk_id: i64,
        _metadata_json: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_all_speakers_with_embeddings(
        &self,
    ) -> BoxFuture<'_, StorageResult<HashMap<i64, Vec<SpeakerEmbeddingV2>>>> {
        Box::pin(async { Ok(HashMap::new()) })
    }

    fn get_speaker_embeddings_v2(
        &self,
        _speaker_id: i64,
    ) -> BoxFuture<'_, StorageResult<Vec<SpeakerEmbeddingRow>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn insert_speaker_embedding_v2(
        &self,
        _speaker_id: i64,
        _embedding: &[f32],
        _model_type: &str,
        _quality_score: f64,
        _audio_chunk_id: Option<i64>,
        _segment_start: Option<f64>,
        _segment_end: Option<f64>,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn prune_speaker_embeddings(
        &self,
        _speaker_id: i64,
        _max_count: i64,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn insert_speaker(&self, _embedding: &[f32]) -> BoxFuture<'_, StorageResult<Speaker>> {
        Box::pin(async {
            Ok(Speaker {
                id: 0,
                name: None,
                metadata: None,
                created_at: Utc::now(),
            })
        })
    }

    fn create_speaker_v2(&self) -> BoxFuture<'_, StorageResult<Speaker>> {
        let pool = self.pool.clone();
        Box::pin(async move {
            let now = Utc::now();
            let created_at_str = now.to_rfc3339();

            let id = sqlx::query_scalar::<_, i64>(
                r#"
                INSERT INTO speakers (name, metadata, embedding_count, last_seen, created_at)
                VALUES (NULL, NULL, 0, ?1, ?1)
                RETURNING id
                "#,
            )
            .bind(&created_at_str)
            .fetch_one(pool.as_ref())
            .await
            .map_err(db_err)?;

            tracing::debug!("Created speaker with DB id: {}", id);

            Ok(Speaker {
                id,
                name: None,
                metadata: None,
                created_at: now,
            })
        })
    }

    fn move_speaker_embeddings(
        &self,
        _from_speaker_id: i64,
        _to_speaker_id: i64,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn update_transcription_speaker(
        &self,
        _from_speaker_id: i64,
        _to_speaker_id: i64,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn delete_speaker(&self, _speaker_id: i64) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn soft_delete_speaker(
        &self,
        _speaker_id: i64,
        _merged_into_id: i64,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_active_speakers(&self) -> BoxFuture<'_, StorageResult<Vec<StoredSpeaker>>> {
        let pool = self.pool.clone();
        Box::pin(async move {
            let rows = sqlx::query_as::<_, (i64, Option<String>, Option<String>, i64, Option<String>)>(
                r#"
                SELECT id, name, metadata, embedding_count, last_seen
                FROM speakers
                WHERE merged_into_id IS NULL
                ORDER BY last_seen DESC
                LIMIT 100
                "#,
            )
            .fetch_all(pool.as_ref())
            .await
            .map_err(db_err)?;

            let speakers = rows
                .into_iter()
                .map(|(id, name, metadata, _embedding_count, _last_seen): (i64, Option<String>, Option<String>, i64, Option<String>)| {
                    StoredSpeaker {
                        id,
                        name,
                        metadata: metadata.and_then(|m| serde_json::from_str(&m).ok()),
                        created_at: Utc::now(), // TODO: Add created_at to query
                    }
                })
                .collect();

            Ok(speakers)
        })
    }

    fn record_merge_event(
        &self,
        _source_speaker_id: i64,
        _target_speaker_id: i64,
        _similarity_score: f32,
        _merge_reason: &str,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn get_merge_history(
        &self,
        _speaker_id: i64,
    ) -> BoxFuture<'_, StorageResult<Vec<MergeHistoryEntry>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn resolve_speaker(&self, speaker_id: i64) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async move { Ok(speaker_id) })
    }

    fn get_diarization_job_by_id(
        &self,
        _job_id: i64,
    ) -> BoxFuture<'_, StorageResult<Option<DiarizationJob>>> {
        Box::pin(async { Ok(None) })
    }

    fn start_diarization_job(&self, _job_id: i64) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn complete_diarization_job(&self, _job_id: i64) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn fail_diarization_job(
        &self,
        _job_id: i64,
        _error_msg: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn update_speaker_for_time_range(
        &self,
        _audio_chunk_id: i64,
        _start_time: f64,
        _end_time: f64,
        _speaker_id: i64,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_next_diarization_job(&self) -> BoxFuture<'_, StorageResult<Option<DiarizationJob>>> {
        Box::pin(async { Ok(None) })
    }

    fn get_diarization_job_by_chunk(
        &self,
        _audio_chunk_id: i64,
    ) -> BoxFuture<'_, StorageResult<Option<DiarizationJob>>> {
        Box::pin(async { Ok(None) })
    }

    fn queue_diarization_job(
        &self,
        _audio_chunk_id: i64,
        _file_path: &str,
        _priority: i32,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn get_audio_transcriptions_for_range(
        &self,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
    ) -> BoxFuture<'_, StorageResult<Vec<AudioTranscription>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn is_unified_transcript_window_processed(
        &self,
        _window_start: DateTime<Utc>,
        _window_end: DateTime<Utc>,
    ) -> BoxFuture<'_, StorageResult<bool>> {
        Box::pin(async { Ok(false) })
    }

    fn insert_unified_transcript_job(
        &self,
        _window_start: DateTime<Utc>,
        _window_end: DateTime<Utc>,
        _priority: i32,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn get_pending_unified_transcript_jobs(
        &self,
        _limit: usize,
    ) -> BoxFuture<'_, StorageResult<Vec<UnifiedTranscriptJob>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn update_unified_transcript_job_status(
        &self,
        _job_id: i64,
        _status: &str,
        _error_msg: Option<&str>,
        _segments_created: Option<i32>,
        _duplicates_removed: Option<i32>,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn delete_unified_segments_for_window(
        &self,
        _window_start: DateTime<Utc>,
        _window_end: DateTime<Utc>,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn insert_unified_transcript_segments_batch(
        &self,
        _segments: &[UnifiedTranscriptSegmentInput],
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn increment_unified_transcript_job_retry(
        &self,
        _job_id: i64,
    ) -> BoxFuture<'_, StorageResult<i32>> {
        Box::pin(async { Ok(0) })
    }

    fn count_diarization_jobs_by_status(&self, _status: &str) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn count_diarization_jobs_completed_today(&self) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn count_diarization_jobs_failed_today(&self) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn log_extraction_skip(
        &self,
        _audio_chunk_id: i64,
        _segment_start: f64,
        _segment_end: f64,
        _event_type: &str,
        _reason: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn log_extraction_success(
        &self,
        _audio_chunk_id: i64,
        _segment_start: f64,
        _segment_end: f64,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn store_conversation_block(
        &self,
        _block: &audiopipe::storage::ConversationBlockInput,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn update_conversation_synopsis(
        &self,
        _conversation_id: &str,
        _synopsis: &str,
        _speaker_synopses: Option<&str>,
        _topic_sections: Option<&str>,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_conversation_blocks(
        &self,
        _start: DateTime<Utc>,
        _end: DateTime<Utc>,
        _limit: usize,
    ) -> BoxFuture<'_, StorageResult<Vec<audiopipe::storage::ConversationBlockOutput>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn get_conversation_block(
        &self,
        _id: &str,
    ) -> BoxFuture<'_, StorageResult<Option<audiopipe::storage::ConversationBlockOutput>>> {
        Box::pin(async { Ok(None) })
    }

    fn store_boundary_decision(
        &self,
        _decision: &audiopipe::storage::BoundaryDecisionInput,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn update_boundary_feedback(
        &self,
        _decision_id: i64,
        _user_override: bool,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_decisions_with_feedback(
        &self,
        _limit: usize,
    ) -> BoxFuture<'_, StorageResult<Vec<audiopipe::storage::StoredBoundaryDecision>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn update_segment_conversation(
        &self,
        _segment_id: &str,
        _conversation_id: &str,
        _topic_label: Option<&str>,
        _topic_section_index: Option<i32>,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn store_threshold_proposal(
        &self,
        _proposal: &audiopipe::storage::ThresholdProposalInput,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn update_proposal_status(
        &self,
        _proposal_id: i64,
        _status: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_pending_proposals(
        &self,
    ) -> BoxFuture<'_, StorageResult<Vec<audiopipe::storage::StoredThresholdProposal>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn store_speaker_block(&self, _block: &SpeakerBlockInput) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn update_speaker_block_summary(
        &self,
        _block_id: &str,
        _summary: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_speaker_blocks_pending_summary(
        &self,
        _limit: usize,
    ) -> BoxFuture<'_, StorageResult<Vec<SpeakerBlockPendingSummary>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn get_speaker_blocks_for_conversation(
        &self,
        _conversation_id: &str,
    ) -> BoxFuture<'_, StorageResult<Vec<SpeakerBlockOutput>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn update_transcription_block_assignment(
        &self,
        _transcription_id: i64,
        _speaker_block_id: &str,
        _conversation_id: Option<&str>,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn get_active_speaker_block(
        &self,
        _device: &str,
    ) -> BoxFuture<'_, StorageResult<Option<SpeakerBlockOutput>>> {
        Box::pin(async { Ok(None) })
    }

    fn finalize_speaker_blocks_for_conversation(
        &self,
        _speaker_block_ids: &[String],
        _conversation_id: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }

    fn link_transcriptions_to_speaker_block(
        &self,
        _first_transcription_id: i64,
        _last_transcription_id: i64,
        _speaker_block_id: &str,
        _conversation_id: Option<&str>,
    ) -> BoxFuture<'_, StorageResult<u64>> {
        Box::pin(async { Ok(0) })
    }

    fn get_speaker_segment_counts(&self) -> BoxFuture<'_, StorageResult<Vec<(i64, i64)>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn get_fragment_speakers(&self, _min_segments: usize) -> BoxFuture<'_, StorageResult<Vec<i64>>> {
        Box::pin(async { Ok(vec![]) })
    }

    fn store_topic_group(&self, group: &TopicGroupInput) -> BoxFuture<'_, StorageResult<()>> {
        let pool = self.pool.clone();
        let id = group.id.clone();
        let conversation_id = group.conversation_id.clone();
        let label = group.label.clone();
        let start_time = group.start_time.to_rfc3339();
        let end_time = group.end_time.to_rfc3339();
        let duration_ms = group.duration_ms;
        let speaker_ids_json = serde_json::to_string(&group.speaker_ids).unwrap_or_default();
        let block_count = group.block_count;
        let status = group.status.clone();

        Box::pin(async move {
            sqlx::query(
                r#"
                INSERT INTO topic_groups (
                    id, conversation_id, label, start_time, end_time,
                    duration_ms, speaker_ids, block_count, status
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                "#,
            )
            .bind(&id)
            .bind(&conversation_id)
            .bind(&label)
            .bind(&start_time)
            .bind(&end_time)
            .bind(duration_ms)
            .bind(&speaker_ids_json)
            .bind(block_count)
            .bind(&status)
            .execute(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(())
        })
    }

    fn update_topic_group(&self, group: &TopicGroupInput) -> BoxFuture<'_, StorageResult<()>> {
        let pool = self.pool.clone();
        let id = group.id.clone();
        let conversation_id = group.conversation_id.clone();
        let label = group.label.clone();
        let end_time = group.end_time.to_rfc3339();
        let duration_ms = group.duration_ms;
        let speaker_ids_json = serde_json::to_string(&group.speaker_ids).unwrap_or_default();
        let block_count = group.block_count;
        let status = group.status.clone();

        Box::pin(async move {
            sqlx::query(
                r#"
                UPDATE topic_groups SET
                    conversation_id = ?2,
                    label = ?3,
                    end_time = ?4,
                    duration_ms = ?5,
                    speaker_ids = ?6,
                    block_count = ?7,
                    status = ?8,
                    updated_at = datetime('now')
                WHERE id = ?1
                "#,
            )
            .bind(&id)
            .bind(&conversation_id)
            .bind(&label)
            .bind(&end_time)
            .bind(duration_ms)
            .bind(&speaker_ids_json)
            .bind(block_count)
            .bind(&status)
            .execute(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(())
        })
    }

    fn add_block_to_topic_group(
        &self,
        topic_group_id: &str,
        speaker_block_id: &str,
        sequence_order: i32,
    ) -> BoxFuture<'_, StorageResult<()>> {
        let pool = self.pool.clone();
        let topic_group_id = topic_group_id.to_string();
        let speaker_block_id = speaker_block_id.to_string();

        Box::pin(async move {
            sqlx::query(
                r#"
                INSERT INTO topic_group_blocks (topic_group_id, speaker_block_id, sequence_order)
                VALUES (?1, ?2, ?3)
                ON CONFLICT(topic_group_id, speaker_block_id) DO UPDATE SET
                    sequence_order = excluded.sequence_order
                "#,
            )
            .bind(&topic_group_id)
            .bind(&speaker_block_id)
            .bind(sequence_order)
            .execute(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(())
        })
    }

    fn get_topic_groups(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: usize,
    ) -> BoxFuture<'_, StorageResult<Vec<StoredTopicGroup>>> {
        let pool = self.pool.clone();
        let start_str = start.to_rfc3339();
        let end_str = end.to_rfc3339();

        Box::pin(async move {
            let rows = sqlx::query_as::<
                _,
                (
                    String,
                    Option<String>,
                    String,
                    String,
                    String,
                    i64,
                    Option<String>,
                    i32,
                    String,
                    String,
                ),
            >(
                r#"
                SELECT id, conversation_id, label, start_time, end_time,
                       duration_ms, speaker_ids, block_count, status, created_at
                FROM topic_groups
                WHERE start_time >= ?1 AND end_time <= ?2
                ORDER BY start_time DESC
                LIMIT ?3
                "#,
            )
            .bind(&start_str)
            .bind(&end_str)
            .bind(limit as i32)
            .fetch_all(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(rows
                .into_iter()
                .map(|r| StoredTopicGroup {
                    id: r.0,
                    conversation_id: r.1,
                    label: r.2,
                    start_time: parse_date(&r.3),
                    end_time: parse_date(&r.4),
                    duration_ms: r.5,
                    speaker_ids: r
                        .6
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default(),
                    block_count: r.7,
                    status: r.8,
                    created_at: parse_date(&r.9),
                })
                .collect())
        })
    }

    fn get_topic_groups_for_conversation(
        &self,
        conversation_id: &str,
    ) -> BoxFuture<'_, StorageResult<Vec<StoredTopicGroup>>> {
        let pool = self.pool.clone();
        let conversation_id = conversation_id.to_string();

        Box::pin(async move {
            let rows = sqlx::query_as::<
                _,
                (
                    String,
                    Option<String>,
                    String,
                    String,
                    String,
                    i64,
                    Option<String>,
                    i32,
                    String,
                    String,
                ),
            >(
                r#"
                SELECT id, conversation_id, label, start_time, end_time,
                       duration_ms, speaker_ids, block_count, status, created_at
                FROM topic_groups
                WHERE conversation_id = ?1
                ORDER BY start_time ASC
                "#,
            )
            .bind(&conversation_id)
            .fetch_all(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(rows
                .into_iter()
                .map(|r| StoredTopicGroup {
                    id: r.0,
                    conversation_id: r.1,
                    label: r.2,
                    start_time: parse_date(&r.3),
                    end_time: parse_date(&r.4),
                    duration_ms: r.5,
                    speaker_ids: r
                        .6
                        .and_then(|s| serde_json::from_str(&s).ok())
                        .unwrap_or_default(),
                    block_count: r.7,
                    status: r.8,
                    created_at: parse_date(&r.9),
                })
                .collect())
        })
    }

    fn get_active_topic_group(&self) -> BoxFuture<'_, StorageResult<Option<StoredTopicGroup>>> {
        let pool = self.pool.clone();

        Box::pin(async move {
            let row = sqlx::query_as::<
                _,
                (
                    String,
                    Option<String>,
                    String,
                    String,
                    String,
                    i64,
                    Option<String>,
                    i32,
                    String,
                    String,
                ),
            >(
                r#"
                SELECT id, conversation_id, label, start_time, end_time,
                       duration_ms, speaker_ids, block_count, status, created_at
                FROM topic_groups
                WHERE status = 'active'
                ORDER BY updated_at DESC
                LIMIT 1
                "#,
            )
            .fetch_optional(pool.as_ref())
            .await
            .map_err(db_err)?;

            Ok(row.map(|r| StoredTopicGroup {
                id: r.0,
                conversation_id: r.1,
                label: r.2,
                start_time: parse_date(&r.3),
                end_time: parse_date(&r.4),
                duration_ms: r.5,
                speaker_ids: r
                    .6
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default(),
                block_count: r.7,
                status: r.8,
                created_at: parse_date(&r.9),
            }))
        })
    }

    fn get_or_insert_audio_chunk(&self, file_path: &str) -> BoxFuture<'_, StorageResult<i64>> {
        let pool = self.pool.clone();
        let file_path = file_path.to_string();
        Box::pin(async move {
            // First try to get existing chunk by file_path
            let existing = sqlx::query_scalar::<_, i64>(
                "SELECT id FROM audio_chunks WHERE file_path = ?1",
            )
            .bind(&file_path)
            .fetch_optional(pool.as_ref())
            .await
            .map_err(db_err)?;

            if let Some(id) = existing {
                return Ok(id);
            }

            // Insert new chunk
            let id = sqlx::query_scalar::<_, i64>(
                r#"
                INSERT INTO audio_chunks (file_path, timestamp)
                VALUES (?1, datetime('now'))
                RETURNING id
                "#,
            )
            .bind(&file_path)
            .fetch_one(pool.as_ref())
            .await
            .map_err(db_err)?;
            Ok(id)
        })
    }

    fn store_transcription(
        &self,
        _input: &TranscriptionInput,
    ) -> BoxFuture<'_, StorageResult<i64>> {
        Box::pin(async { Ok(0) })
    }

    fn link_transcription_to_speaker_block(
        &self,
        _transcription_id: i64,
        _speaker_block_id: &str,
    ) -> BoxFuture<'_, StorageResult<()>> {
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_storage_types_compile() {
        // Verify the storage adapter compiles with audiopipe types
    }
}
