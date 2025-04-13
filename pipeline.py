#!/usr/bin/env python3

import sys
from RealtimeSTT import AudioToTextRecorder
import os
from mistralai import Mistral
import argparse
import subprocess
from uuid import uuid4

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"
client = Mistral(api_key=api_key)


def process_audio_command(recorder):
    recorder.start()
    input("Press Enter to stop recording...")
    recorder.stop()
    transcribed_text = recorder.text()
    print("Transcription: ", transcribed_text)

    prompt_simple_task = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that maps natural language instructions to either 'stack balls' or 'rest'. "
                "If the user's command is about stacking, piling, or building a tower with balls, output 'stack balls'. "
                "If the user says 'bowl' or 'bowl', assume he meant 'balls' or 'ball'."
                "For any other command, output 'rest'. Return only one of these phrases, no explanations.\n\n"
                "Here are some examples:\n"
                'Input: "Can you pile the balls up like a tower?"\n'
                'Output: "stack balls"\n\n'
                'Input: "Move the balls out of the way."\n'
                'Output: "rest"\n\n'
                'Input: "Toss the balls across the room."\n'
                'Output: "rest"'
            ),
        },
        {"role": "user", "content": f'Input: "{transcribed_text}"\nOutput:'},
    ]

    prompt_multi_tasks = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that maps natural language instructions to one of three robot tasks: "
                "'stack balls', 'push balls', or 'throw balls'. Return only one of these phrases, no explanations. If the user's command is not about stacking, pushing, or throwing balls, return 'rest'. if the user says 'bowl' or 'bowl', assume he meant 'balls' or 'ball'. \n\n"
                "Here are some examples:\n"
                'Input: "Can you pile the balls up like a tower?"\n'
                'Output: "stack balls"\n\n'
                'Input: "Move the balls out of the way."\n'
                'Output: "push balls"\n\n'
                'Input: "Toss the balls across the room."\n'
                'Output: "throw balls"'
            ),
        },
        {"role": "user", "content": f'Input: "{transcribed_text}"\nOutput:'},
    ]

    # Use transcribed text as input for Mistral API
    chat_response = client.chat.complete(model=model, messages=prompt_simple_task)

    result = chat_response.choices[0].message.content
    print("Robot task: ", result)

    if "rest" in result.strip().lower():
        try:
            subprocess.run(["ffplay", "-nodisp", "-autoexit", "hal9000.mp3"])
        except Exception as e:
            print(f"Error playing audio file: {e}")
    else:
        # Run the robot control script with the specified parameters
        try:
            subprocess.run([
                "python",
                "/home/andrej/hackathon/lerobot/lerobot/scripts/control_robot.py",
                "--robot.type=so100",
                "--control.type=record",
                "--control.fps=30",
                "--control.single_task=Stack the balls on top of each other.",
                f"--control.repo_id=AndrejOrsula/eval_so100_tripple_ball_stacking_{uuid4()}",
                '--control.tags=["tutorial"]',
                "--control.warmup_time_s=0.5",
                "--control.episode_time_s=120",
                "--control.reset_time_s=1",
                "--control.num_episodes=1",
                "--control.push_to_hub=false",
                "--control.policy.path=/home/andrej/hackathon/act/so100_tripple_ball_stacking_100k/pretrained_model",
            ])  # so100_tripple_ball_stacking_100k
        except Exception as e:
            print(f"Error running robot control script: {e}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech to robot task converter")
    parser.add_argument(
        "--continuous", "-c", action="store_true", help="Enable continuous listening mode (default: False)"
    )
    args = parser.parse_args()

    try:
        if args.continuous:
            print("Starting continuous listening mode (Press Ctrl+C to exit)...")
            while True:
                print("\nReady to listen for a new command...")
                with AudioToTextRecorder() as recorder:
                    process_audio_command(recorder)
                print("\nPress Ctrl+C to exit or wait for next recording...")
        else:
            # Single recording mode (original behavior)
            with AudioToTextRecorder() as recorder:
                process_audio_command(recorder)
            sys.exit(0)

    except KeyboardInterrupt:
        print("\nExiting program...")
        sys.exit(0)
