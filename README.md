# Voice Control

## Overview
A voice-activated assistant that:
1. Listens for trigger words from `settings/triggerwords.txt`
2. Processes follow-up commands from `settings/commands.txt`
3. Executes corresponding system commands
4. Supports dynamic arguments in commands

## Setup Guide
1. Clone this repo
2. Download [faster whisper](https://huggingface.co/Systran) models
3. Place in `models/[model_size]`
4. Configure your:
   - `settings/triggerwords.txt`
   - `settings/commands.txt`
   - `settings/config.txt`
5. run `main.py`

## How to Use
1. Speak any trigger word
2. Program will listen for a command
3. If command is found programm will match it against `settings/commands.txt`
4. And then programm will execute corresponding system command

## File Specifications

### settings/triggerwords.txt
- **Format**: Comma-separated phrases
- **Example**:
```
voice control,voicecontrol,voices control,voicescontrol
```
### settings/commands.txt
- **Format**: `activation_phrase::system_command`
- **Arguments**: Use `@1`, `@2`, etc.
- **Examples**:
```
open calculator::calc
```
```
google @1 in the browser::python open_url.py "https://www.google.com/search?q=@1"
```

- **additional notes on command creation**:
   - No duplicate activation phrases
   - No command that's a substring of another
   - Arguments must be separated by text
   - Phrase cannot end with argument
   - Unique text before first argument

### settings/сonfig.txt

| Parameter                    | Valid Options                          | Description |
|----------------------------|----------------------------------------|-------------|
| **Language Settings**      |                                        |             |
| `language_code`            | ISO 639-1 codes (en,de,fr,ru,etc.)     | Speech recognition language |
|                            |                                        |             |
| **Debug Settings**         |                                        |             |
| `debug_prints`             | `True`/`False`                         | Enable general debug messages |
| `debug_time_prints`        | `True`/`False`                | Enable timing measurements for each processing step |
|                            |                                        |             |
| **Model Settings**         |                                        |             |
| `torch_device`             | `cuda`/`cpu`                           | Processing device (GPU/CPU) |
| `model_name`               | `tiny`,`small`,`medium`,`large-v2`,`large-v3`,`large-v3-turbo`    | Whisper model size (larger = more accurate but slower) |
| `compute_type`             | `int8`,`float16`,`float32`         | Computation precision (affects speed/accuracy) |
|                            |                                        |             |
| **Audio Settings**         |                                        |             |
| `audio_device_input_id`          | Device ID or `-1`             | Microphone input device (-1 shows selection dialog) |
| `audio_device_output_id`          | Device ID or `-1`             | Headphones/Speakers output device (-1 shows selection dialog) |
| `sample_rate`              | Typically `16000` or `44100`           | Audio sampling rate in Hz |
| `audio_channels`           | `1` (mono) or `2` (stereo)             | Audio channels |
| `trigger_word_wait_time`   | Number of cycles (-1 for infinite)| How long to wait for command after trigger word, each cycle takes chunk_size_seconds-2\*overlap_size_seconds rounded to the nearest audio_block_size_seconds (0.7 seconds for default config) |
| `chunk_size_seconds`       | any float      | Audio chunk size for processing, the bigger the longer activation phrases could be |
| `overlap_size_seconds`     | any float ( < than chunk_size_seconds/2 )                       | Relative overlap between audio chunks |
| `audio_block_size_seconds` | any float                        | Audio capture block size |
| `push_to_talk_enabled` | `True`/`False`                       | Enable push-to-talk (programm will react only if you are pressing the `push_to_talk_button`) |
| `push_to_talk_toggle_button` | key or combination of keys separated by +                        | if pressed programm will react to audio (only if `push_to_talk_enabled` is set to true) |
| `play_sound_triggerword` | `True`/`False`                          | play audio if triggerword is found |
| `play_sound_command` | `True`/`False`                        | play audio if command is executed |
| `typing_mode_toggle_button` | key or combination of keys separated by +                        | if pressed programm will switch typing mode (all audio will be ryped by your keyboard) |


## Troubleshooting
- Check microphone settings/permissions
- Verify model files exist
- Enable debug prints
- Check file formatting

## Common errors
`Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!`
DLLs can be downloaded [here](https://github.com/Purfview/whisper-standalone-win/releases/tag/libs)
 
`cudaErrorNoKernelImageForDevice: no kernel image is available for execution on the device`
Your GPU is too old for ctranslate2 version 4
You could try [this](https://github.com/m-bain/whisperX/issues/794#issuecomment-2103963143) workaround




