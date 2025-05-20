import sys
import time
import keyboard

def main():
    if len(sys.argv) < 2:
        print("Usage: python execute_keystrokes.py \"command1;command2;...\"")
        sys.exit(1)
    
    # Split commands and remove any empty strings
    commands = [cmd.strip() for cmd in sys.argv[1].split(';') if cmd.strip()]
    
    for i, cmd in enumerate(commands):
        # Send the keystroke combination
        keyboard.send(cmd)
        # Wait 0.5 seconds after each command except the last one
        if i < len(commands) - 1:
            time.sleep(0.5)
    
if __name__ == "__main__":
    main()