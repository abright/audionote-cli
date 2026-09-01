# audionote-cli

Did this for fun over a couple nights one weekend.
It records audio from the default microphone and transcribes it using Whisper, printing the output to the terminal.

Accuracy seems quite good if you speak clearly to it in a quiet room - I haven't done anything to improve signal quality such as background noise reduction, or dynamic range compression to boost quiet voices.

Those would be next steps if turning this from a weekend experiment project into a genuinely useful tool.

## Basic architecture & how it works
Audio recording is continuous but uses a simple RMS method to detect activity and buffers that to send to a separate processing thread after detecting a period of silence.
The processing thread resamples the incoming audio buffer and feeds it to Whisper for transcription. That output is then sent to the main thread for display.

Communication between threads uses `crossbeam-channel`.

Audio capture uses `cpal` with processing handled by `rubato` for resampling and `whisper-rs` for inference.

It's a simple pipeline: audio thread (managed by `cpal`) → buffered speech → resampling/whisper thread → transcribed text → main thread.

## Usage
Download Whisper tiny or base models (maybe larger ones will work, they haven't been tested) and store in `models/` directory.
Set the correct model filename (look for `MODEL_PATH`) in the code and `cargo run`


### Notes on AI usage
- Written with partial AI assistance, running Gemma 4 26B A4B locally.
