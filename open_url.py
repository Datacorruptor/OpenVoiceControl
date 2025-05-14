
#!/usr/bin/env python3

import webbrowser
import sys

def open_url(url: str):
    """
    Opens the specified URL in the default web browser.
    
    Args:
        url (str): The URL to open
    """
    try:
        # Open URL in a new browser window
        webbrowser.open(url, new=2)
        print(f"Successfully opened {url} in your default browser.")
    except Exception as e:
        print(f"An error occurred while trying to open {url}: {e}")

if __name__ == "__main__":
    # Check if URL is provided as command-line argument
    if len(sys.argv) < 2:
        print("Usage: python open_url.py <url>")
        print("Example: python open_url.py example.com")
        sys.exit(1)
    
    # Get URL from command-line argument
    url = sys.argv[1]
        
    open_url(url)