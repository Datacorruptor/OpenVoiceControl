import os
import sys

import requests
import random
import time

'''
this script communicates with a qiqi bot(https://www.botqiqi.com/) to play music in your channel
to support other bots you need to rewrite this script for your bot
'''

DISCORD_TOKEN = os.environ["discord_token"] #your discord token

SERVER_ID = "352498268548104203" #place your server id where you want to play music
TEXT_CHANEL_ID = "509093278700929025" #channel id where you able to communicate with the bot

def generate_random_hex(length=32):
    hex_chars = '0123456789abcdef'
    return ''.join(random.choices(hex_chars, k=length))


def generate_random_number(start=0, end=10000000000000):
    return random.randint(start, end)

def find_and_play_track(querry):
    token = DISCORD_TOKEN
    headers = {
        "Authorization" : token
    }
    session = generate_random_hex()
    nonce = generate_random_number()

    data = {
        "payload_json":'{"type":2,"application_id":"927363789089833021","guild_id":"'+SERVER_ID+'","channel_id":"'+TEXT_CHANEL_ID+'","session_id":"'+session+'","data":{"version":"1336495382414491702","id":"986769964159619163","name":"play","type":1,"options":[{"type":3,"name":"search","value":"'+querry+'"}],"application_command":{"id":"986769964159619163","type":1,"application_id":"927363789089833021","version":"1336495382414491702","name":"play","description":"Play music from YouTube, Bilibili, Netease, and Spotify","description_default":"Play music from YouTube, Bilibili, Netease, and Spotify","options":[{"type":3,"name":"search","description":"Keyword (YouTube)| url (YoutTube & Bilibili & Netease) | for searching music","description_default":"Keyword (YouTube)| url (YoutTube & Bilibili & Netease) | for searching music","required":true,"description_localized":"url (Soundcloud & Bilibili & Netease & Pengu) | Keyword (Soundcloud) for searching music","name_localized":"search"}],"dm_permission":true,"integration_types":[0],"global_popularity_rank":1,"description_localized":"Play music from Soundcloud, Bilibili, Netease, and Pengu","name_localized":"play"},"attachments":[]},"nonce":"'+str(nonce)+'","analytics_location":"slash_ui"}'
    }
    # be aware of discord api changes
    url = "https://discord.com/api/v9/interactions"

    requests.post(url,headers=headers,data=data)

    # be aware of discord api changes
    url = f"https://discord.com/api/v9/channels/{TEXT_CHANEL_ID}/messages"

    time.sleep(5)

    nonce = generate_random_number()
    data = {"mobile_network_type":"unknown","content":"1","nonce":str(nonce),"tts":False,"flags":0}
    requests.post(url,headers=headers,data=data)


if __name__ == "__main__":
    # Check if URL is provided as command-line argument
    if len(sys.argv) < 2:
        print('Usage: python find_and_play_track_discord_qiqi.py "<querry>"')
        print('Example: python find_and_play_track_discord_qiqi.py "stairway to heaven"')
        sys.exit(1)

    # Get URL from command-line argument
    querry = sys.argv[1]

    find_and_play_track(querry)