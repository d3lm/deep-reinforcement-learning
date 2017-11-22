import os

def get_time_difference(start, end):
  return (end - start)*1000.

def clear_screen(lines=500):
  if os.name == "posix":
    # Unix/Linux/MacOS/BSD/etc
    os.system('clear')
  elif os.name in ("nt", "dos", "ce"):
    # DOS/Windows
    os.system('cls')
  else:
    # Fallback for other operating systems
    print('\n' * numlines)
