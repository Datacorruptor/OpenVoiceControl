import subprocess
import sys
import os
dll_path = os.path.join(os.path.dirname(__file__), "dlls")
os.add_dll_directory(dll_path)

os.environ["PATH"] = dll_path+os.pathsep+os.environ["PATH"]

import sounddevice as sd
import soundfile as sf
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

import time
import threading
from transliterate import translit
from Levenshtein import distance
import traceback
import string


import re
from difflib import SequenceMatcher

def word_inclusion(words,phrase):
    return " "+words+" " in " "+phrase+" "

def find_longest_intersection_suffix_array(str1, str2):
    if not str1 or not str2:
        return ""
    str1 = str1.replace("ё", "е")
    str2 = str2.replace("ё", "е")

    combined = str1 + "#" + str2 + "$"
    n1, n2 = len(str1), len(str2)

    suffixes = [(combined[i:], i) for i in range(len(combined))]
    suffixes.sort()

    max_len = 0
    result = ""

    for i in range(len(suffixes) - 1):
        suffix1, pos1 = suffixes[i]
        suffix2, pos2 = suffixes[i + 1]

        if (pos1 < n1 and pos2 > n1) or (pos1 > n1 and pos2 < n1):
            lcp_len = 0
            min_len = min(len(suffix1), len(suffix2))

            for j in range(min_len):
                if suffix1[j] == suffix2[j] and suffix1[j] not in "#$":
                    lcp_len += 1
                else:
                    break

            if lcp_len > max_len:
                max_len = lcp_len
                result = suffix1[:lcp_len]

    return result


def remove_punctuation(text):
    return ''.join(char for char in text if char not in string.punctuation or char in "-")

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
    config["push_to_talk_enabled"] = config["push_to_talk_enabled"] == "True"
    config["play_sound_triggerword"] = config["play_sound_triggerword"] == "True"
    config["play_sound_command"] = config["play_sound_command"] == "True"

    config["push_to_talk_toggle_button"] = config["push_to_talk_toggle_button"].split("+")
    config["typing_mode_toggle_button"] = config["typing_mode_toggle_button"].split("+")

    print(config["debug_prints"])

    if config["debug_prints"]:
        print("loaded config", config)
    
    return config


def configure_input_audio_device_stream(config):
    if config["debug_prints"]:
        # List available audio devices
        print("Available input audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}")

    # Select microphone device
    input_device_id = int(config["audio_device_input_id"])
    if input_device_id == -1:
        input_device_id = int(input("Enter the device ID for your microphone: "))
    if config["debug_prints"]:
        print("selected device",input_device_id)

    if config["debug_prints"]:
        # List available audio devices
        print("Available output audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                print(f"{i}: {device['name']}")

    # Select microphone device
    output_device_id = int(config["audio_device_output_id"])
    if output_device_id == -1:
        output_device_id = int(input("Enter the device ID for your speakers/headphones: "))
    if config["debug_prints"]:
        print("selected device", output_device_id)

    sd.default.device = input_device_id, output_device_id

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
        device=input_device_id
    )

    return stream


def parse_triggers_and_commands(config):

    triggers = open("settings\\triggerwords.txt",encoding="utf-8").read().split('\n')[0].split(",")
    if config["debug_prints"]:
        print("current triggers: ", triggers)

    command_lines = open("settings\\commands.txt",encoding="utf-8").read().split('\n')
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
        if tr in text:
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

def handle_commands(config, txt, init_phrases, init2full , commands, sounds, triggered ,last_cmd):
    for phr in init_phrases:
        if phr not in txt:
            continue

        full_phrase = init2full[phr]

        if full_phrase in txt:
            triggered = False
            if config["debug_prints"]:
                print(phr, commands[phr])

            if are_commands_different_enough(last_cmd, commands[phr], config["language_code"]):
                if config["debug_prints"]:
                    print(f"executing command {commands[phr]}")
                if config["play_sound_command"]:
                    sd.play(*sounds["ok"])
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
                if config["debug_prints"]:
                    print(f"executing command {full_command}")
                if config["play_sound_command"]:
                    sd.play(*sounds["ok"])
                execute_command_in_cmd(full_command)
            last_cmd = full_command
            break
    else:
        last_cmd = ""

    return triggered, last_cmd

def segments_intersect(a1, a2, b1, b2):
    if a2 < b1 or b2 < a1:
        return False
    else:
        return True

def segments_distance(a1, a2, b1, b2):
    a1, a2 = min(a1, a2), max(a1, a2)
    b1, b2 = min(b1, b2), max(b1, b2)

    overlap_start = max(a1, b1)
    overlap_end = min(a2, b2)

    if overlap_start <= overlap_end:
        overlap_length = overlap_end - overlap_start
        return -overlap_length
    else:
        if a2 < b1:
            return b1 - a2
        else:
            return a1 - b2

def handle_audio_segments(config, detection_data, trigger_word_wait, init_phrases, init2full , commands, sounds, triggered ,last_cmd, typing_mode_data):
    t1 = None
    model_detection_result, aligned_model_detection_result, audio_size = detection_data
    prev_text, typed_text, typed_words, prev_words, candidate_words, global_time, typing_mode = typing_mode_data
    print(global_time)
    #print(aligned_model_detection_result)

    if typing_mode:
        words = []
        for s in aligned_model_detection_result["segments"]:
            words.extend(s["words"])

        #print(words)
        words = [
            {
                "word": remove_punctuation(w["word"].strip().lower()),
                "start": global_time+w["start"],
                "end": global_time+w["end"],
                "score": w["score"]
            }
            for w in words if w["start"] if w["start"] > 0.5 and w["end"] < audio_size / float(config["sample_rate"]) - 0.5 ]
        print(words)

        #words = [w for w in words if w["score"] > 0.1]
        to_delete = []
        for w1 in candidate_words:
            for i in range(len(typed_words)):
                w2 = typed_words[i]
                if segments_distance(w1["start"], w1["end"], w2["start"], w2["end"]) <0:
                    to_delete.append(w1)
        for w in to_delete:
            try:
                candidate_words.remove(w)
            except:
                print(w)
                print(candidate_words)



        new_words = []
        for w1 in words[:]:
            for i in range(len(typed_words)):
                w2 = typed_words[i]
                segm_intersection = segments_distance(w1["start"],w1["end"],w2["start"],w2["end"])
                if segm_intersection < 0:
                    break

                if segm_intersection < 0.5:
                    if w1["word"] == w2["word"]:
                        break

                if segm_intersection < 1.0:
                    if distance(w1["word"], w2["word"]) <= min(len(w1["word"]), len(w2["word"])) // 10:
                        break

            else:

                for w3 in candidate_words:
                    if w1["word"] == w3["word"] and abs(w1["start"]-w3["start"]) < 1 and abs(w1["end"]-w3["end"]) < 1:
                        new_words.append(w1)
                        break
                else:
                    candidate_words.append(w1)

        print("new words", new_words)


        ttt = " ".join([w["word"] for w in new_words])

        if len(ttt) != 0:
            if len(typed_words) != 0:
                ttt = " "+ttt

            print("new text", ttt)
            threading.Thread(target=lambda: keyboard.write(ttt)).start()
            typed_words.extend(new_words)

        prev_words = words


    for segment in model_detection_result["segments"]:
        text = remove_punctuation(segment["text"].strip().lower())

        if not text:
            #ttt = prev_text.replace(find_longest_intersection_suffix_array(prev_text, typed_text), "")
            #threading.Thread(target=lambda: keyboard.write(ttt)).start()
            #print("typed", ttt)

            if typing_mode:
                ttt = " ".join([w["word"] for w in prev_words[-1:]])

                if len(ttt) != 0:
                    if len(typed_words) != 0:
                        ttt = " " + ttt
                    threading.Thread(target=lambda: keyboard.write(ttt)).start()

            last_cmd = ""
            typed_text = ""
            prev_text = ""
            typed_words = []
            prev_words = []
            candidate_words = []
            continue

        '''        print("prev_text",prev_text)
        print("text", text)
        intersection = find_longest_intersection_suffix_array(prev_text, text)
        print("intersection", intersection)

        print(len(intersection),word_inclusion(intersection,text),word_inclusion(intersection,prev_text))
        if len(intersection) > 3 and word_inclusion(intersection,text) and word_inclusion(intersection,prev_text):
            old_typed_text = typed_text
            print("old_text", old_typed_text)
            already_typed = find_longest_intersection_suffix_array(old_typed_text, intersection)
            print("already_typed", already_typed)

            if not word_inclusion(already_typed,old_typed_text):
                already_typed = ""

            if not word_inclusion(already_typed,intersection):
                already_typed = ""

            if old_typed_text.endswith(already_typed) and intersection.startswith(already_typed) or already_typed == "":
                ttt = intersection.replace(already_typed,"")
                #threading.Thread(target=lambda: keyboard.write(ttt)).start()
                print("typed new", ttt)
                typed_text = old_typed_text.strip() + " " + ttt.strip()
                print("full typed", typed_text)

        prev_text = text'''



        if config["debug_prints"]:
            print(f"{trigger_word_wait} > {text}")

        if trigger_word_wait == 0:
            triggered = False
            last_cmd = ""
        trigger_word_wait -= 1

        if config["debug_time_prints"]:
            t1 = time.time()

        if check_trigger_word(config, text, triggers):
            if not triggered and config['play_sound_triggerword']:
                sd.play(*sounds["yes"])

            triggered = True
            trigger_word_wait = trigger_word_wait_time

        if config["debug_time_prints"]:
            t2 = time.time()
            print("trigger finding time", t2 - t1)

        if config["debug_time_prints"]:
            t1 = time.time()

        if not triggered:
            continue

        triggered, last_cmd = handle_commands(config, text, init_phrases, init2full, commands, sounds, triggered, last_cmd)

        if config["debug_time_prints"]:
            t2 = time.time()
            print("command finding time", t2 - t1)



    return triggered, last_cmd, trigger_word_wait, prev_text, typed_text, typed_words, prev_words, candidate_words


def load_sounds():
    # Load "yes" sound
    data, fs = sf.read('settings/yes.mp3', dtype='float32')
    print(f"'yes' shape: {data.shape}, channels: {data.shape[1] if len(data.shape) > 1 else 1}")  # Debug
    yes_sound = (data, fs)

    # Load "ok" sound
    data, fs = sf.read('settings/ok.mp3', dtype='float32')
    print(f"'ok' shape: {data.shape}, channels: {data.shape[1] if len(data.shape) > 1 else 1}")  # Debug
    ok_sound = (data, fs)

    sounds = {"yes": yes_sound, "ok": ok_sound}

    # List all devices
    devices = sd.query_devices()
    # Check default output device
    default_out = sd.default.device[1]  # Output device index
    print(
        f"\nDefault Output Device: {devices[default_out]['name']} (Channels: {devices[default_out]['max_output_channels']})")

    return sounds

def  whisperx_load_align_model(config):
    align_model, metadata = whisperx.load_align_model(language_code=config['language_code'], device=config['torch_device'])
    return align_model, metadata


if __name__ == "__main__":
    config = parse_config("settings\\config.txt")
    stream = configure_input_audio_device_stream(config)
    triggers, commands, phrases, init_phrases, init2full = parse_triggers_and_commands(config)
    sounds = load_sounds()
    model = load_whisperx_model(config)
    align_model, metadata = whisperx_load_align_model(config)

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
        prev_text = ""
        typed_text = ""
        typed_words = []
        prev_words = []
        candidate_words = []
        main_loop_start_time = time.time()
        typing_mode_toggle_button_state = False
        typing_mode_toggle_button_state_prev = False
        typing_mode = False
        ptt_mode_toggle_button_state = False
        ptt_mode_toggle_button_state_prev = False
        ptt_mode = False
        while True:
            global_time = time.time() - main_loop_start_time

            print("typing mode", typing_mode)
            print("push-to-talk mode", ptt_mode)

            all_pressed = True
            for k in config["typing_mode_toggle_button"]:
                if not keyboard.is_pressed(k):
                    all_pressed = False
                    break

            if all_pressed:
                typing_mode_toggle_button_state = True
                if typing_mode_toggle_button_state_prev == False:
                    typing_mode = not typing_mode
            else:
                typing_mode_toggle_button_state = False

            typing_mode_toggle_button_state_prev = typing_mode_toggle_button_state

            all_pressed = True
            for k in config["push_to_talk_toggle_button"]:
                if not keyboard.is_pressed(k):
                    all_pressed = False
                    break

            if all_pressed:
                ptt_mode_toggle_button_state = True
                if ptt_mode_toggle_button_state_prev == False:
                    ptt_mode = not ptt_mode
            else:
                ptt_mode_toggle_button_state = False

            ptt_mode_toggle_button_state_prev = ptt_mode_toggle_button_state

            if not ptt_mode and config["push_to_talk_enabled"]:
                if config["debug_time_prints"]:
                    print("sleeping, waiting for push-to-talk")

                #ttt = prev_text.replace(find_longest_intersection_suffix_array(prev_text, typed_text), "")
                #threading.Thread(target=lambda: keyboard.write(ttt)).start()
                #print("typed", ttt)

                if typing_mode:

                    ttt = " ".join([w["word"] for w in prev_words[-1:]])

                    if len(ttt) != 0:
                        if len(typed_words) != 0:
                            ttt = " " + ttt
                        threading.Thread(target=lambda: keyboard.write(ttt)).start()


                last_cmd = ""
                prev_text = ""
                typed_text = ""
                typed_words = []
                time.sleep(0.2)
                prev_overlap = np.empty((0, 1))
                #clearing the audio queue
                try:
                    while True:
                        audio_queue.get(False)
                except queue.Empty:
                    pass

                if not typing_mode:
                    continue
                else:
                    print("typing mode active, overriding push-to-talk")


            tt1 = time.time()
            required_new_samples = int((float(config["chunk_size_seconds"]) - float(config["overlap_size_seconds"])) * int(config["sample_rate"]))

            audio = prev_overlap.copy()

            if audio_queue.empty():
                time.sleep(0.1)
                continue


            while not audio_queue.empty():
                audio = np.vstack([audio, audio_queue.get()])

            print("audio shape",audio.shape)

            chunk_cutoff = int(float(config["chunk_size_seconds"]) * int(config["sample_rate"]))
            overlap_cutoff = int(float(config["overlap_size_seconds"]) * int(config["sample_rate"]))
            audio = audio[:min(chunk_cutoff,audio.shape[0]-1)]
            prev_overlap = audio[-min(overlap_cutoff,audio.shape[0]-1):]

            # Convert to FP32 tensor
            print("audio shape", audio.shape)
            audio_tensor = torch.from_numpy(audio.flatten().astype(np.float32))
            print("audio_tensor shape", audio_tensor.shape)

            try:

                t1 = None
                if config["debug_time_prints"]:
                    t1 = time.time()

                model_detection_result = model.transcribe(audio_tensor.numpy(), batch_size=4)
                aligned_model_detection_result = whisperx.align(
                    model_detection_result["segments"],
                    align_model,
                    metadata,
                    audio_tensor.numpy(),
                    config['torch_device']
                )
                audio_size = audio.shape[0]
                detection_data = (model_detection_result, aligned_model_detection_result, audio_size)

                if (len(model_detection_result['segments']) == 0
                        or "DimaTorzok" in model_detection_result['segments'][0]['text']
                        or "Продолжение следует..." in model_detection_result['segments'][0]['text']):
                    if config["debug_prints"]:
                        print("No speech...",end="", flush=True)

                    #ttt = prev_text.replace(find_longest_intersection_suffix_array(prev_text, typed_text), "")
                    #threading.Thread(target=lambda: keyboard.write(ttt)).start()
                    #print("typed", ttt)

                    if typing_mode:
                        ttt = " ".join([w["word"] for w in prev_words[-1:]])

                        if len(ttt) != 0:
                            if len(typed_words) != 0:
                                ttt = " " + ttt
                            threading.Thread(target=lambda: keyboard.write(ttt)).start()

                    last_cmd = ""
                    prev_text = ""
                    typed_text = ""
                    typed_words = []
                    prev_words = []
                    candidate_words = []
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


                typing_mode_data = (prev_text, typed_text, typed_words, prev_words, candidate_words, global_time, typing_mode)

                triggered, last_cmd, trigger_word_wait, prev_text, typed_text, typed_words, prev_words , candidate_words= handle_audio_segments(
                    config,
                    detection_data,
                    trigger_word_wait,
                    init_phrases,
                    init2full,
                    commands,
                    sounds,
                    triggered,
                    last_cmd,
                    typing_mode_data
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