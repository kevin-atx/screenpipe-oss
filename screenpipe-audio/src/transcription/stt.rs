use crate::core::device::AudioDevice;
use crate::core::engine::AudioTranscriptionEngine;
use crate::speaker::embedding::EmbeddingExtractor;
use crate::speaker::embedding_manager::EmbeddingManager;
use crate::speaker::prepare_segments;
use crate::speaker::segment::SpeechSegment;
use crate::transcription::deepgram::batch::transcribe_with_deepgram;
use crate::transcription::whisper::batch::process_with_whisper;
use crate::utils::audio::resample;
use crate::utils::ffmpeg::{get_new_file_path, write_audio_to_file};
use crate::vad::VadEngine;
use anyhow::Result;
#[cfg(target_os = "macos")]
use objc::rc::autoreleasepool;
use screenpipe_core::Language;
use std::path::PathBuf;
use std::{
    sync::Arc,
    sync::Mutex as StdMutex,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::Mutex;
use tracing::{error, info};
use whisper_rs::WhisperContext;

use crate::{AudioInput, TranscriptionResult};

// Audiopipe imports (when pro-audio feature is enabled)
#[cfg(feature = "pro-audio")]
use audiopipe::{
    storage::ScreenpipeStorage, AudioChunkInput, AudiopipeConfig, ChunkProcessingInput,
    ChunkProcessor, DeviceType as AudiopipeDeviceType,
};
#[cfg(feature = "pro-audio")]
use chrono::Utc;
#[cfg(feature = "pro-audio")]
use screenpipe_db::DatabaseManager;
#[cfg(feature = "pro-audio")]
use serde_json;
#[cfg(feature = "pro-audio")]
use tokio::sync::OnceCell;

/// Global audiopipe processor (initialized once, reused for all chunks)
#[cfg(feature = "pro-audio")]
static AUDIOPIPE_PROCESSOR: OnceCell<ChunkProcessor> = OnceCell::const_new();

/// Get or initialize the audiopipe processor
/// The `db` parameter is only used on the first call (OnceCell init).
#[cfg(feature = "pro-audio")]
async fn get_audiopipe_processor(_db: &Arc<DatabaseManager>) -> Result<&'static ChunkProcessor> {
    AUDIOPIPE_PROCESSOR
        .get_or_try_init(|| async {
            info!("Initializing audiopipe ChunkProcessor...");

            // Try loading config from ~/.screenpipe/audiopipe.toml, fall back to default
            let mut config = {
                let config_path = dirs::home_dir()
                    .map(|h| h.join(".screenpipe").join("audiopipe.toml"))
                    .filter(|p| p.exists());

                if let Some(path) = config_path {
                    match AudiopipeConfig::from_file(&path) {
                        Ok(cfg) => {
                            info!("Loaded audiopipe config from {:?}", path);
                            cfg
                        }
                        Err(e) => {
                            error!("Failed to load audiopipe config from {:?}: {}, using defaults", path, e);
                            AudiopipeConfig::default()
                        }
                    }
                } else {
                    AudiopipeConfig::default()
                }
            };

            // Override LLM settings from screenpipe's AI configuration (store.bin)
            if let Some(data_dir) = dirs::data_dir() {
                let store_path = data_dir.join("screenpipe").join("store.bin");
                if let Ok(content) = std::fs::read_to_string(&store_path) {
                    if let Ok(store) = serde_json::from_str::<serde_json::Value>(&content) {
                        if let Some(model) = store.get("aiModel").and_then(|v| v.as_str()) {
                            info!("Using LLM model from screenpipe settings: {}", model);
                            config.conversation.llm_model = model.to_string();
                        }
                        if let Some(url) = store.get("aiUrl").and_then(|v| v.as_str()) {
                            // Strip /v1 suffix — audiopipe uses Ollama native API (/api/generate)
                            let base_url = url.trim_end_matches("/v1").to_string();
                            config.conversation.llm_base_url = base_url;
                        }
                    }
                }
            }

            // Create storage adapter from screenpipe's db path
            let db_path = dirs::home_dir()
                .map(|h| h.join(".screenpipe").join("db.sqlite"))
                .expect("home dir not found");
            let storage = ScreenpipeStorage::new(
                db_path.to_str().expect("invalid db path"),
            )
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create audiopipe storage: {}", e))?;

            let processor = ChunkProcessor::with_storage(
                config,
                Arc::new(storage),
            )
            .await?;

            info!("Audiopipe ChunkProcessor initialized with database storage");
            Ok(processor)
        })
        .await
}

/// Proactively initialize the audiopipe processor at startup.
/// Call this before processing audio to avoid cold-start delay on the first chunk.
#[cfg(feature = "pro-audio")]
pub async fn init_audiopipe(db: &Arc<DatabaseManager>) -> Result<()> {
    get_audiopipe_processor(db).await?;
    Ok(())
}

pub const SAMPLE_RATE: u32 = 16000;

#[allow(clippy::too_many_arguments)]
pub async fn stt_sync(
    audio: &[f32],
    sample_rate: u32,
    device: &str,
    audio_transcription_engine: Arc<AudioTranscriptionEngine>,
    deepgram_api_key: Option<String>,
    languages: Vec<Language>,
    whisper_context: Arc<WhisperContext>,
) -> Result<String> {
    let audio = audio.to_vec();

    let device = device.to_string();

    stt(
        &audio,
        sample_rate,
        &device,
        audio_transcription_engine,
        deepgram_api_key,
        languages,
        whisper_context,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
pub async fn stt(
    audio: &[f32],
    sample_rate: u32,
    device: &str,
    audio_transcription_engine: Arc<AudioTranscriptionEngine>,
    deepgram_api_key: Option<String>,
    languages: Vec<Language>,
    whisper_context: Arc<WhisperContext>,
) -> Result<String> {
    let transcription: Result<String> =
        if audio_transcription_engine == AudioTranscriptionEngine::Deepgram.into() {
            // Deepgram implementation
            let api_key = deepgram_api_key.unwrap_or_default();

            match transcribe_with_deepgram(&api_key, audio, device, sample_rate, languages.clone())
                .await
            {
                Ok(transcription) => Ok(transcription),
                Err(e) => {
                    error!(
                        "device: {}, deepgram transcription failed, falling back to Whisper: {:?}",
                        device, e
                    );
                    // Fallback to Whisper
                    process_with_whisper(audio, languages.clone(), whisper_context).await
                }
            }
        } else {
            // Existing Whisper implementation
            process_with_whisper(audio, languages, whisper_context).await
        };

    transcription
}

#[allow(clippy::too_many_arguments)]
pub async fn process_audio_input(
    audio: AudioInput,
    vad_engine: Arc<Mutex<Box<dyn VadEngine + Send>>>,
    segmentation_model_path: PathBuf,
    embedding_manager: EmbeddingManager,
    embedding_extractor: Arc<StdMutex<EmbeddingExtractor>>,
    output_path: &PathBuf,
    audio_transcription_engine: Arc<AudioTranscriptionEngine>,
    deepgram_api_key: Option<String>,
    languages: Vec<Language>,
    output_sender: &crossbeam::channel::Sender<TranscriptionResult>,
    whisper_context: Arc<WhisperContext>,
) -> Result<()> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    let audio_data = if audio.sample_rate != SAMPLE_RATE {
        resample(audio.data.as_ref(), audio.sample_rate, SAMPLE_RATE)?
    } else {
        audio.data.as_ref().to_vec()
    };

    let audio = AudioInput {
        data: Arc::new(audio_data.clone()),
        sample_rate: SAMPLE_RATE,
        ..audio
    };

    let (mut segments, speech_ratio_ok) = prepare_segments(
        &audio_data,
        vad_engine,
        &segmentation_model_path,
        embedding_manager,
        embedding_extractor,
        &audio.device.to_string(),
    )
    .await?;

    if !speech_ratio_ok {
        return Ok(());
    }

    let new_file_path = get_new_file_path(&audio.device.to_string(), output_path);

    if let Err(e) = write_audio_to_file(
        &audio.data.to_vec(),
        audio.sample_rate,
        &PathBuf::from(&new_file_path),
        false,
    ) {
        error!("Error writing audio to file: {:?}", e);
    }

    while let Some(segment) = segments.recv().await {
        let path = new_file_path.clone();
        let transcription_result = if cfg!(target_os = "macos") {
            #[cfg(target_os = "macos")]
            {
                let timestamp = timestamp + segment.start.round() as u64;
                autoreleasepool(|| {
                    run_stt(
                        segment,
                        audio.device.clone(),
                        audio_transcription_engine.clone(),
                        deepgram_api_key.clone(),
                        languages.clone(),
                        path,
                        timestamp,
                        whisper_context.clone(),
                    )
                })
                .await?
            }
            #[cfg(not(target_os = "macos"))]
            {
                unreachable!("This code should not be reached on non-macOS platforms")
            }
        } else {
            run_stt(
                segment,
                audio.device.clone(),
                audio_transcription_engine.clone(),
                deepgram_api_key.clone(),
                languages.clone(),
                path,
                timestamp,
                whisper_context.clone(),
            )
            .await?
        };

        if output_sender.send(transcription_result).is_err() {
            break;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub async fn run_stt(
    segment: SpeechSegment,
    device: Arc<AudioDevice>,
    audio_transcription_engine: Arc<AudioTranscriptionEngine>,
    deepgram_api_key: Option<String>,
    languages: Vec<Language>,
    path: String,
    timestamp: u64,
    whisper_context: Arc<WhisperContext>,
) -> Result<TranscriptionResult> {
    let audio = segment.samples.clone();
    let sample_rate = segment.sample_rate;
    match stt_sync(
        &audio,
        sample_rate,
        &device.to_string(),
        audio_transcription_engine.clone(),
        deepgram_api_key.clone(),
        languages.clone(),
        whisper_context,
    )
    .await
    {
        Ok(transcription) => Ok(TranscriptionResult {
            input: AudioInput {
                data: Arc::new(audio),
                sample_rate,
                channels: 1,
                device: device.clone(),
            },
            transcription: Some(transcription),
            path,
            timestamp,
            error: None,
            speaker_embedding: segment.embedding.clone(),
            start_time: segment.start,
            end_time: segment.end,
            speaker_id: None, // OSS path - will be matched via embedding
            pipeline_metadata: None, // OSS path - no pipeline metadata
        }),
        Err(e) => {
            error!("STT error for input {}: {:?}", device, e);
            Ok(TranscriptionResult {
                input: AudioInput {
                    data: Arc::new(segment.samples),
                    sample_rate: segment.sample_rate,
                    channels: 1,
                    device: device.clone(),
                },
                transcription: None,
                path,
                timestamp,
                error: Some(e.to_string()),
                speaker_embedding: Vec::new(),
                start_time: segment.start,
                end_time: segment.end,
                speaker_id: None,
                pipeline_metadata: None,
            })
        }
    }
}

// ============================================================================
// AUDIOPIPE PROCESSING (Pro feature - replaces OSS VAD/transcription)
// ============================================================================

/// Process audio using audiopipe (Pro audio processing)
///
/// This bypasses the OSS VAD (which has issues with speech_frames: 0) and uses
/// audiopipe's Pyannote-based VAD, quality assessment, and Whisper transcription.
///
/// Key differences from OSS processing:
/// - Always saves audio to file (no VAD gating)
/// - Uses Pyannote for VAD (more accurate than Silero)
/// - Includes quality assessment and preprocessing
/// - Better speaker diarization
#[cfg(feature = "pro-audio")]
#[allow(clippy::too_many_arguments)]
pub async fn process_audio_input_with_audiopipe(
    audio: AudioInput,
    output_path: &PathBuf,
    output_sender: &crossbeam::channel::Sender<TranscriptionResult>,
    db: Arc<DatabaseManager>,
) -> Result<()> {
    use crate::core::device::DeviceType;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_secs();

    // Resample to 16kHz if needed
    let audio_data = if audio.sample_rate != SAMPLE_RATE {
        resample(audio.data.as_ref(), audio.sample_rate, SAMPLE_RATE)?
    } else {
        audio.data.as_ref().to_vec()
    };

    let audio = AudioInput {
        data: Arc::new(audio_data.clone()),
        sample_rate: SAMPLE_RATE,
        ..audio
    };

    // ALWAYS save audio to file (bypass VAD gate - this was the OSS bug)
    let new_file_path = get_new_file_path(&audio.device.to_string(), output_path);
    info!(
        "Audiopipe: saving audio to file (bypassing VAD): {}",
        new_file_path
    );

    if let Err(e) = write_audio_to_file(
        &audio.data.to_vec(),
        audio.sample_rate,
        &PathBuf::from(&new_file_path),
        false,
    ) {
        error!("Error writing audio to file: {:?}", e);
        return Err(e.into());
    }

    // Get or initialize audiopipe processor
    let processor = match get_audiopipe_processor(&db).await {
        Ok(p) => p,
        Err(e) => {
            error!("Failed to initialize audiopipe processor: {:?}", e);
            return Err(e);
        }
    };

    // Map device type
    let device_type = match audio.device.device_type {
        DeviceType::Input => AudiopipeDeviceType::Input,
        DeviceType::Output => AudiopipeDeviceType::Output,
    };

    // Create audiopipe input
    let chunk_input = AudioChunkInput {
        chunk_path: PathBuf::from(&new_file_path),
        device_name: audio.device.to_string(),
        device_type,
        timestamp: Utc::now(),
        duration_ms: 30000, // 30 second chunks
    };

    let processing_input = ChunkProcessingInput {
        chunk: chunk_input,
        known_speakers: vec![],
    };

    // Process with audiopipe
    info!("Audiopipe: processing audio chunk from {}", audio.device);
    let output = match processor.process(processing_input).await {
        Ok(output) => output,
        Err(e) => {
            error!("Audiopipe processing failed: {:?}", e);
            // Send error result
            let _ = output_sender.send(TranscriptionResult {
                input: audio.clone(),
                transcription: None,
                path: new_file_path,
                timestamp,
                error: Some(format!("Audiopipe error: {}", e)),
                speaker_embedding: Vec::new(),
                start_time: 0.0,
                end_time: 30.0,
                speaker_id: None,
                pipeline_metadata: None,
            });
            return Err(e.into());
        }
    };

    // Map audiopipe output to TranscriptionResult(s)
    // Audiopipe may produce multiple segments with different speakers
    let transcription_text = &output.transcription.primary_text;

    if transcription_text.is_empty() {
        info!(
            "Audiopipe: no transcription for {} (VAD detected no speech)",
            audio.device
        );
        return Ok(());
    }

    info!(
        "Audiopipe: transcribed {} chars from {} ({} speakers detected)",
        transcription_text.len(),
        audio.device,
        output.speakers.len()
    );

    // Find the first speaker with a valid database ID (numeric string like "289")
    // Skip speakers with session-local IDs like "speaker_001" or "unknown"
    let matched_speaker = output
        .speakers
        .iter()
        .find(|s| s.speaker_id.parse::<i64>().is_ok());

    let (audiopipe_speaker_id, speaker_embedding) = if let Some(speaker) = matched_speaker {
        let id = speaker.speaker_id.parse::<i64>().ok();
        let embedding = speaker
            .new_embeddings
            .first()
            .map(|e| e.embedding.clone())
            .unwrap_or_default();
        info!(
            "Audiopipe: using matched speaker ID {} for {} (confidence: {:?})",
            speaker.speaker_id, audio.device, speaker.match_confidence
        );
        (id, embedding)
    } else {
        // No cross-session match found - log all speaker IDs for debugging
        let speaker_ids: Vec<&str> = output.speakers.iter().map(|s| s.speaker_id.as_str()).collect();
        info!(
            "Audiopipe: no cross-session match for {} (session speakers: {:?})",
            audio.device, speaker_ids
        );
        let embedding = output
            .speakers
            .first()
            .and_then(|s| s.new_embeddings.first())
            .map(|e| e.embedding.clone())
            .unwrap_or_default();
        (None, embedding)
    };

    // Build pipeline metadata JSON for database storage
    // This captures quality, hallucination scores, and confidence for posthumous analysis
    let pipeline_metadata = {
        let metadata = serde_json::json!({
            "quality": {
                "score": output.quality.score,
                "snr_db": output.quality.snr_db,
                "clipping_ratio": output.quality.clipping_ratio,
                "volume_dbfs": output.quality.volume_dbfs,
            },
            "hallucination": output.hallucination.as_ref().map(|h| serde_json::json!({
                "is_hallucination": h.is_hallucination,
                "score": h.score,
                "adjusted_score": h.adjusted_score,
                "matched_patterns": h.matched_patterns,
                "words_per_second": h.words_per_second,
            })),
            "transcription": {
                "outcome_type": output.transcription.outcome_type,
                "decision_reason": output.transcription.decision_reason,
                "confidence": {
                    "raw": output.transcription.confidence.raw,
                    "calibrated": output.transcription.confidence.calibrated,
                    "calibration_method": output.transcription.confidence.calibration_method,
                    "snr_adjusted": output.transcription.confidence.snr_adjusted,
                }
            },
            "validation": output.validation_metrics.as_ref().map(|v| serde_json::json!({
                "alignment_quality": v.alignment_quality,
                "orphan_word_ratio": v.orphan_word_ratio,
                "orphan_word_count": v.orphan_word_count,
                "word_count": v.word_count,
                "vad_hallucination_score": v.vad_hallucination_score,
            })),
            "vad": {
                "speech_ratio": output.vad.speech_ratio,
                "segment_count": output.vad.vad_segments.len(),
                "speaker_count": output.speakers.len(),
            },
        });
        Some(metadata.to_string())
    };

    // For now, send a single TranscriptionResult with the full transcription
    // TODO: split by speaker segments for better speaker attribution
    let result = TranscriptionResult {
        input: audio.clone(),
        transcription: Some(transcription_text.clone()),
        path: new_file_path,
        timestamp,
        error: None,
        speaker_embedding,
        start_time: 0.0,
        end_time: output.duration_ms as f64 / 1000.0,
        // Use audiopipe's matched speaker_id directly (bypasses OSS re-matching)
        speaker_id: audiopipe_speaker_id,
        pipeline_metadata,
    };

    if output_sender.send(result).is_err() {
        error!("Failed to send transcription result");
    }

    Ok(())
}

/// Fallback when pro-audio feature is not enabled
#[cfg(not(feature = "pro-audio"))]
pub async fn process_audio_input_with_audiopipe(
    _audio: AudioInput,
    _output_path: &PathBuf,
    _output_sender: &crossbeam::channel::Sender<TranscriptionResult>,
    _db: Arc<screenpipe_db::DatabaseManager>,
) -> Result<()> {
    Err(anyhow::anyhow!(
        "Audiopipe processing requires the 'pro-audio' feature to be enabled"
    ))
}
