import subprocess
import sys
import sounddevice as sd
import numpy as np
import torch
from transformers.image_utils import *
import whisperx.diarize
import whisperx.asr
import whisperx
import lightning_fabric
import faster_whisper
import asteroid_filterbanks
import pyannote
import pyannote.audio.models
import pyannote.audio.models.segmentation
import pyannote.audio.models.blocks
import pyannote.audio.utils
import pyannote.audio.utils.receptive_field
import pyannote.audio.utils.params
import ctranslate2
import sklearn.calibration
import speechbrain.dataio
import speechbrain.utils
import speechbrain
import keyboard

import queue
import os
import time
import threading
from transliterate import translit
from Levenshtein import distance

import traceback

def parse_config(path):
    config_text = open(path,encoding="utf-8").read()
    config = {}
    for line in config_text.split("\n"):
        if line == "":
            continue
        param, value = line.split(":")[:2]
        config[param] = value

    config["debug_prints"] = config["debug_prints"] == "True"
    config["debug_time_prints"] = config["debug_time_prints"] == "True"
    config["push_to_talk"] = config["push_to_talk"] == "True"

    config["push_to_talk_button"] = config["push_to_talk_button"].split("+")

    print(config["debug_prints"])

    if config["debug_prints"]:
        print("loaded config", config)
    
    return config


def configure_input_audio_device_stream(config):
    if config["debug_prints"]:
        # List available audio devices
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}")

    # Select microphone device
    device_id = int(config["audio_device_id"])
    if device_id == -1:
        device_id = int(input("Enter the device ID for your microphone: "))
    if config["debug_prints"]:
        print("selected device",device_id)
    sd.default.device = device_id

    def audio_callback(indata, frames, time, status):
        """Audio input callback"""
        if status:
            print(f"Audio input error: {status}")
        audio_queue.put(indata.copy())

    # Start streaming with device selection
    stream = sd.InputStream(
        samplerate=int(config["sample_rate"]),
        blocksize=int(int(config["sample_rate"]) * float(config["audio_block_size_seconds"])),  # 0.5 second blocks
        channels=int(config["audio_channels"]),
        callback=audio_callback,
        device=device_id
    )

    return stream


def parse_triggers_and_commands(config):

    triggers = open("triggerwords.txt",encoding="utf-8").read().split('\n')[0].split(",")
    if config["debug_prints"]:
        print("current triggers: ", triggers)

    command_lines = open("commands.txt",encoding="utf-8").read().split('\n')
    commands = {}
    phrases = []
    init_phrases = []
    init2full = {}
    double_phrases_start = {}
    for l in command_lines:
        if l == "":
            continue

        phrase,command = l.split("::")[:2]
        commands[phrase] = command
        phrases.append(phrase)
        init_phrase = phrase.split("@")[0]
        init_phrases.append(init_phrase)
        init2full[init_phrase] = phrase
        phrase_words = phrase.split()
        double_word = (phrase_words[0],phrase_words[1])
        if double_word in double_phrases_start:
            double_phrases_start[double_word].append(phrase)
        else:
            double_phrases_start[double_word] = [phrase]
        if config["debug_prints"]:
            print("added command: '",command,"' on activation phrase '",phrase,"'")

    return triggers, commands, phrases, init_phrases, init2full


def load_whisperx_model(config):
    # Initialize WhisperX
    model = whisperx.load_model(
        "models/"+config["model_name"],
        config["torch_device"],
        compute_type=config["compute_type"],
        language=config["language_code"],
        local_files_only=True
    )
    return model

def check_trigger_word(config, text, triggers):
    for tr in triggers:
        if tr in text.lower():
            if config["debug_prints"]:
                print("TRIGGER WORD")
            return True
    return False

def are_commands_different_enough(c1,c2,language):
    if c1 == c2:
        if config["debug_prints"]:
            print("exactly the same command, dont execute")
        return False

    translit1 = translit(c1, language)
    translit2 = translit(c2, language)
    if config["debug_prints"]:
        print("checking transliteration (", translit1,"?=?", translit2, ") distance:", distance(translit1, translit2))
    if distance(translit1, translit2) <= min(len(translit1), len(translit2)) // 10:
        return False
    return True

def prepare_command(command):
    for k in os.environ:
        if k in str(command).upper():
            command = command.replace("%"+k+"%", os.environ[k])
    return command

def execute_command_in_cmd(command):
    full_command = command
    prepared_command = prepare_command(full_command)
    debug_command = 'cmd /c "' + prepared_command + '"'
    threading.Thread(target=lambda: subprocess.run(debug_command, timeout=30)).start()

def replace_arguments_in_command(config, txt, full_phrase, full_command, parts):
    for part_index in range(0, len(parts) - 1):
        arg_start = txt.find(parts[part_index]) + len(parts[part_index])
        arg_end = txt.find(parts[part_index + 1])
        arg_value = txt[arg_start:arg_end]
        if config["debug_prints"]:
            print(f"replacing @{part_index + 1} with {arg_value}")
        full_phrase = full_phrase.replace(f"@{part_index + 1}", arg_value)
        full_command = full_command.replace(f"@{part_index + 1}", arg_value)
    return full_phrase, full_command

def handle_commands(config, txt, init_phrases, init2full , commands, triggered ,last_cmd):
    for phr in init_phrases:
        if phr not in txt:
            continue

        full_phrase = init2full[phr]

        if full_phrase in txt:
            triggered = False
            if config["debug_prints"]:
                print(phr, commands[phr])

            if are_commands_different_enough(last_cmd, commands[phr], config["language_code"]):
                execute_command_in_cmd(commands[phr])
            last_cmd = commands[phr]
            break
        else:

            cnt = str(full_phrase).count("@")
            if cnt == 0:
                continue

            parts_phrase = full_phrase
            for i in range(1, cnt + 1):
                parts_phrase = full_phrase.replace(f"@{i}", "@@")
            parts = parts_phrase.split("@@")

            if not all([i in txt for i in parts]):
                continue

            if config["debug_prints"]:
                print(f"complex command, all parts in place")

            full_command = commands[full_phrase]

            full_phrase, full_command = replace_arguments_in_command(config, txt, full_phrase, full_command, parts)

            if config["debug_prints"]:
                print(full_phrase, full_command)

            if are_commands_different_enough(last_cmd, full_command, config["language_code"]):
                execute_command_in_cmd(full_command)
            last_cmd = full_command
            break
    else:
        last_cmd = ""

    return triggered, last_cmd


def handle_audio_segments(config, result, trigger_word_wait, init_phrases, init2full , commands, triggered ,last_cmd):
    t1 = None
    for segment in result["segments"]:
        text = segment["text"].strip()

        if not text:
            last_cmd = ""
            continue

        if config["debug_prints"]:
            print(f"{trigger_word_wait} > {text}")

        if trigger_word_wait == 0:
            triggered = False
            last_cmd = ""
        trigger_word_wait -= 1

        if config["debug_time_prints"]:
            t1 = time.time()

        if check_trigger_word(config, text, triggers):
            triggered = True
            trigger_word_wait = trigger_word_wait_time

        if config["debug_time_prints"]:
            t2 = time.time()
            print("trigger finding time", t2 - t1)

        if config["debug_time_prints"]:
            t1 = time.time()

        if not triggered:
            continue

        txt = text.lower()
        triggered, last_cmd = handle_commands(config, txt, init_phrases, init2full, commands, triggered, last_cmd)

        if config["debug_time_prints"]:
            t2 = time.time()
            print("command finding time", t2 - t1)

    return triggered, last_cmd, trigger_word_wait

if __name__ == "__main__":
    config = parse_config("config.txt")
    stream = configure_input_audio_device_stream(config)
    triggers, commands, phrases, init_phrases, init2full = parse_triggers_and_commands(config)
    model = load_whisperx_model(config)

    # Audio buffer setup
    audio_queue = queue.Queue()
    prev_overlap = np.empty((0, 1))

    if config["debug_prints"]:
        print("\nListening... (press Ctrl+C to stop)")
    stream.start()
    triggered = False
    trigger_word_wait_time = int(config["trigger_word_wait_time"])
    trigger_word_wait = trigger_word_wait_time

    try:
        pr_word = ""
        last_cmd = ""
        while True:

            all_pressed = True
            for k in config["push_to_talk_button"]:
                if not keyboard.is_pressed(k):
                    all_pressed = False
                    break

            if not all_pressed and config["push_to_talk"]:
                if config["debug_time_prints"]:
                    print("sleeping, waiting for push-to-talk")
                last_cmd = ""
                time.sleep(0.5)
                prev_overlap = np.empty((0, 1))
                continue

            tt1 = time.time()
            required_new_samples = int((float(config["chunk_size_seconds"]) - float(config["overlap_size_seconds"])) * int(config["sample_rate"]))

            audio = prev_overlap.copy()

            if audio_queue.empty():
                time.sleep(0.1)
                continue


            while not audio_queue.empty():
                audio = np.vstack([audio, audio_queue.get()])

            chunk_cutoff = int(float(config["chunk_size_seconds"]) * int(config["sample_rate"]))
            overlap_cutoff = int(float(config["overlap_size_seconds"]) * int(config["sample_rate"]))
            audio = audio[:min(chunk_cutoff,audio.shape[0]-1)]
            prev_overlap = audio[-min(overlap_cutoff,audio.shape[0]-1):]

            # Convert to FP32 tensor
            audio_tensor = torch.from_numpy(audio.flatten().astype(np.float32))

            try:



                t1 = None
                if config["debug_time_prints"]:
                    t1 = time.time()
                result = model.transcribe(audio_tensor.numpy(), batch_size=4)

                if (len(result['segments']) == 0
                        or "DimaTorzok" in result['segments'][0]['text']
                        or "Продолжение следует..." in result['segments'][0]['text']):
                    if config["debug_prints"]:
                        print("No speech...",end="", flush=True)
                    last_cmd = ""
                    continue

                if config["debug_time_prints"] or config["debug_prints"]:
                    print("===============================================")

                if config["debug_time_prints"]:
                    t2 = time.time()
                    print("model inference time", t2-t1)

                if config['debug_prints']:
                    #print("results:",result)
                    pass

                if config['debug_prints']:
                    print("last command: ",last_cmd)




                triggered, last_cmd, trigger_word_wait = handle_audio_segments(
                    config,
                    result,
                    trigger_word_wait,
                    init_phrases,
                    init2full,
                    commands,
                    triggered,
                    last_cmd
                )



            except Exception as e:
                print(f"Processing error: {e}")
                print(traceback.print_exc())
                continue

            tt2 = time.time()
            print("cycle time", tt2 - tt1)

    except KeyboardInterrupt:
        if config["debug_prints"]:
            print("\nStopping...")
        stream.stop()
        stream.close()