use anyhow::{bail, Context, Result};
use cpal::traits::*;
use crossbeam_channel::unbounded;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{self, ClearType},
};
use rubato::{audioadapter_buffers::direct::InterleavedSlice, Fft, FixedSync, Resampler};
use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const THRESHOLD: f32 = 0.01; // RMS value, below this we consider it 'silence'
const SILENCE_LIMIT: usize = 50; // number of buffers of 'silence' to trigger our processing
const MODEL_PATH: &str = "models/ggml-tiny.bin"; // path for the whisper model we'll load

enum EventMessage {
    Error(String), // error message (eg: resampler runs out of buffer space) - may not be fatal
    Info(String),  // info message (eg: "processed buffer of 12434 samples")
    Transcription(String), // transcription output, we'll print in a different colour
}

// audio resampler context to wrap up init and buffer processing
struct AudioResampler {
    resampler: Fft<f32>,
    input_fs: usize,
    output_fs: usize,
}

impl AudioResampler {
    pub fn new(input_fs: usize, output_fs: usize) -> Result<Self> {
        let resampler = Fft::new(
            input_fs,
            output_fs,
            input_fs / 50, // 960 samples at 48khz (seems to hit the callback this often so we'll get multiples of it)
            1,
            FixedSync::Input,
        )?;
        Ok(Self {
            resampler,
            input_fs,
            output_fs,
        })
    }

    pub fn process(&mut self, input: Vec<f32>) -> Result<Vec<f32>> {
        // Mono conversion (De-interleave: L, R, L, R -> L, L, L) - microphone recording is typically a single channel
        let mono_input: Vec<f32> = input.chunks(2).map(|chunk| chunk[0]).collect();

        let chunk_size = self.input_fs / 50;
        // we'll add two chunks worth of audio (about 100ms worth) as the buffer size it calculates here is often a bit short
        let expected_len = chunk_size * 2 + (mono_input.len() * self.output_fs) / self.input_fs;
        // allocate an output buffer
        let mut output = vec![0.0; expected_len];

        let input_adapter = InterleavedSlice::new(&mono_input, 1, mono_input.len())
            .context("failed to create input adapter")?;
        let mut output_adapter = InterleavedSlice::new_mut(&mut output, 1, expected_len)
            .context("failed to create output adapter")?;

        // not all results here are complete failures - if we run out of output buffer, for example, the output we get is still usable
        let (_, _) = self
            .resampler
            .process_all_into_buffer(&input_adapter, &mut output_adapter, mono_input.len(), None)
            .context("resampling error")?;

        Ok(output)
    }
}

fn main() -> Result<()> {
    // this should suppress the whisper and GGML logging that otherwise spams the terminal
    whisper_rs::install_logging_hooks();

    // Set up terminal and terminal scroll region
    terminal::enable_raw_mode()?;
    let (_, rows) = terminal::size()?;
    let mut stdout = std::io::stdout();

    // Lock scrolling to rows 1 through (rows - 1).
    // The bottom-most row (rows) will never scroll automatically.
    let scroll_region_end = rows - 1;
    execute!(
        stdout,
        terminal::Clear(ClearType::All),
        Print(format!("\x1b[1;{}r", scroll_region_end))
    )?;

    // audio setup - default host, input device, configuration
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .context("Failed to find a default input device.")?;
    let config = device
        .default_input_config()
        .context("Failed to get default input config.")?;

    let desc = device
        .description()
        .context("Failed to get audio device description")?
        .name()
        .to_string();

    execute!(
        stdout,
        Print(format!("Input device: {}", desc)),
        Print(format!(
            "Config: {} Hz, {} channels",
            config.sample_rate(),
            config.channels()
        )),
    )?;

    let input_fs = config.sample_rate() as usize;

    // channel to send 'completed' buffers (as we're going to wait for silence to send them)
    let (tx, rx) = crossbeam_channel::unbounded::<Vec<f32>>();

    // silence detection state (note: the audio thread will claim these)
    let is_speaking = Arc::new(AtomicBool::new(false));
    let silence_counter = Arc::new(AtomicUsize::new(0));
    let volume = Arc::new(AtomicUsize::new(0)); // volume as 0-100 integer

    // capture buffer is awkward living out here, since we're using it inside the callback (which runs in a separate thread!)
    let capture_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

    // speaking/silence/volume clones that the ui will use
    let speaking_ui = is_speaking.clone();
    let silence_ui = silence_counter.clone();
    let volume_ui = volume.clone();

    let stream = device
        .build_input_stream(
            config.into(),
            // note: this callback runs in a separate thread
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if data.is_empty() {
                    return;
                }

                // calculate RMS value of the buffer
                let len = data.len() as f32;
                let sum: f32 = data.iter().map(|&x| x * x).sum();
                let rms = (sum / len).sqrt();

                // store a ui-friendly version for the status display
                let vol = (rms * 100.0).min(100.0) as usize;
                volume.store(vol, Ordering::SeqCst);

                // are we speaking? (or at least, making a lot of noise)
                if rms > THRESHOLD {
                    is_speaking.store(true, Ordering::SeqCst);
                    silence_counter.store(0, Ordering::SeqCst);
                    let mut buf = capture_buffer.lock().unwrap();
                    buf.extend_from_slice(data);
                } else {
                    // all quiet, check if we've gone long enough to trigger processing
                    if is_speaking.load(Ordering::SeqCst) {
                        let count = silence_counter.fetch_add(1, Ordering::SeqCst) + 1;
                        if count >= SILENCE_LIMIT {
                            is_speaking.store(false, Ordering::SeqCst);
                            let mut buf = capture_buffer.lock().unwrap();
                            // drain the buffer into a new vec and send the new one
                            let _ = tx.send(buf.drain(..).collect());
                        }
                    }
                }
            },
            |err| eprintln!("\nAn error occurred while reading audio: {}", err),
            None,
        )
        .context("Failed to build input stream.")?;

    let (event_tx, event_rx) = unbounded::<EventMessage>();
    let whisper_thread = thread::spawn(move || -> Result<_> {
        // whisper loading is heavy, do it first in case it fails
        event_tx
            .send(EventMessage::Info(format!(
                "[Worker] loading whisper model: {}\n",
                MODEL_PATH
            )))
            .context("Failed to send model loading status event")?;
        let ctx_params = WhisperContextParameters {
            use_gpu: true,
            flash_attn: false,
            ..Default::default()
        };
        let ctx = match WhisperContext::new_with_params(MODEL_PATH, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                let _ = event_tx.send(EventMessage::Error(format!("Failed to load model: {}", e)));
                bail!(format!("Failed to load model: {}", e));
            }
        };

        // resampler takes whatever the microphone gives us and spits out the 16khz that whisper likes
        let mut resampler =
            AudioResampler::new(input_fs, 16000).context("Failed to create audio resampler.")?;

        let mut state = ctx
            .create_state()
            .context("Failed to create Whisper state")?;

        event_tx
            .send(EventMessage::Info(
                "[Worker] Transcription thread started. Waiting for audio...\n".to_string(),
            ))
            .context("Failed to send start event")?;

        // chill until a new buffer is ready (or the channel breaks, which happens when we drop the audio stream)
        while let Ok(buffer) = rx.recv() {
            let sample_count = buffer.len();

            let output = resampler
                .process(buffer)
                .context("Failed to resample buffer")?;

            event_tx
                .send(EventMessage::Info(format!(
                    "\n[Worker] 🎙️ Received chunk: {} samples. Processed into {} samples\n",
                    sample_count,
                    output.len()
                )))
                .context("Failed to send status event")?;

            let mut params = FullParams::new(SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            });
            params.set_language(None); // auto-detect language

            if let Err(e) = state.full(params, output.as_slice()) {
                event_tx
                    .send(EventMessage::Error(format!(
                        "Failed to process Whisper: {}",
                        e
                    )))
                    .context("failed to send Whisper error event")?;
            }

            for segment in state.as_iter() {
                event_tx
                    .send(EventMessage::Transcription(format!(
                        "[{} - {}]: {}\n",
                        segment.start_timestamp(),
                        segment.end_timestamp(),
                        segment
                    )))
                    .context("failed to send transcription event")?;
            }

            event_tx
                .send(EventMessage::Info(
                    "[Worker] ✅ Finished processing chunk.\n".to_string(),
                ))
                .context("failed to send status event")?;
        }

        event_tx
            .send(EventMessage::Info(
                "[Worker] Thread shutting down.\n".to_string(),
            ))
            .context("Failed to send stop event")?;

        Ok(())
    });

    // start the audio stream
    stream.play().context("Failed to start audio stream.")?;

    // keep main thread alive and handle our ui events
    loop {
        let (_, current_rows) = terminal::size()?;

        // hit 'q' to exit as ctrl-c and other keys aren't processed the usual way in raw terminal mode
        if event::poll(Duration::from_millis(50))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }

        let volume = volume_ui.load(Ordering::SeqCst);
        let volume_bar: String = "#".repeat(volume / 2);
        let active = speaking_ui.load(Ordering::SeqCst);
        let silence_bar: String = "=".repeat(silence_ui.load(Ordering::SeqCst) / 5);

        let status = if active {
            format!(
                "\rVolume: [{:<50}] [🎙  ] [{:<10}] | press q to exit",
                volume_bar, silence_bar
            )
        } else {
            format!(
                "\rVolume: [{:<50}] [   ]              | press q to exit",
                volume_bar
            )
        };

        execute!(
            stdout,
            cursor::MoveTo(0, current_rows - 1),
            ResetColor,
            terminal::Clear(ClearType::CurrentLine),
            Print(status),
        )?;

        execute!(stdout, cursor::MoveTo(0, current_rows - 2))?;
        if let Ok(event) = event_rx.try_recv() {
            match event {
                EventMessage::Info(msg) => {
                    // we don't change colour here, but we could
                    execute!(
                        stdout,
                        //SetForegroundColor(Color::Blue),
                        Print(msg),
                    )?;
                }
                EventMessage::Transcription(msg) => {
                    execute!(stdout, SetForegroundColor(Color::Green), Print(msg),)?;
                }
                EventMessage::Error(msg) => {
                    execute!(stdout, SetForegroundColor(Color::Red), Print(msg),)?;
                }
            }
        }

        thread::sleep(Duration::from_millis(50));
    }

    let (_, final_rows) = terminal::size()?;
    execute!(
        stdout,
        ResetColor,
        Print("\x1b[r"), // unlock/reset scrolling region
        cursor::MoveTo(0, final_rows - 1),
        terminal::Clear(ClearType::CurrentLine)
    )?;
    terminal::disable_raw_mode()?;

    stream.pause().context("Failed to pause audio stream")?;
    // feels dirty dropping the stream like this, but it will cause the whisper thread to shut down too
    drop(stream);

    // join
    whisper_thread
        .join()
        .expect("Whisper thread join failed")
        .context("Whisper thread result")?;

    Ok(())
}
