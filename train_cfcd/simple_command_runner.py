#!/usr/bin/env python3
"""
Simple Command Runner

This script allows you to list commands in a text file or YAML file and run them sequentially
with proper logging, timing, and error handling.

Usage:
    python simple_command_runner.py --commands my_experiments.txt
    python simple_command_runner.py --commands my_experiments.txt --continue_on_error
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Run a list of commands sequentially')
    parser.add_argument('--commands', type=str, required=True,
                        help='Path to file containing commands (txt, yaml, or yml)')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue running other commands if one fails')
    parser.add_argument('--log_dir', type=str, default='command_logs',
                        help='Directory to store command logs')
    return parser.parse_args()

def load_commands(commands_path):
    """Load commands from text file or YAML file."""
    commands_path = Path(commands_path)
    
    if not commands_path.exists():
        raise FileNotFoundError(f"Commands file not found: {commands_path}")
    
    commands = []
    # Text file format
    with open(commands_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith('#'):
            commands.append({
                'command': line,
                'name': f'Command {i}',
            })
    
    return commands

def log_message(message, log_file=None):
    """Log message to console and optionally to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    print(formatted_message)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted_message + '\n')

def run_command(cmd_info, log_file=None, continue_on_error=False):
    """Run a single command."""
    
    command = cmd_info['command']
    name = cmd_info['name']
    
    log_message(f"Starting: {name}", log_file)
    log_message(f"  Command: {command}", log_file)
    
    start_time = time.time()
    
    try:
        # Change working directory if specified
        original_cwd = os.getcwd()
        
        # Run the command
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=True,
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        log_message(f"✓ Command completed successfully in {duration:.2f}s: {name}", log_file)
        
        # Log stdout if present
        if result.stdout.strip():
            log_message("STDOUT:", log_file)
            log_message(result.stdout.strip(), log_file)
        
        return True
        
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        duration = end_time - start_time
        
        log_message(f"✗ Command failed after {duration:.2f}s: {name}", log_file)
        log_message(f"Error code: {e.returncode}", log_file)
        
        if e.stdout and e.stdout.strip():
            log_message("STDOUT:", log_file)
            log_message(e.stdout.strip(), log_file)
        if e.stderr and e.stderr.strip():
            log_message("STDERR:", log_file)
            log_message(e.stderr.strip(), log_file)
        
        if not continue_on_error:
            raise
        
        return False
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        
        log_message(f"💥 Command failed with exception after {duration:.2f}s: {name}", log_file)
        log_message(f"Exception: {str(e)}", log_file)
        
        if not continue_on_error:
            raise
        
        return False
    
def main():
    args = parse_args()
    
    # Load commands
    try:
        commands = load_commands(args.commands)
    except Exception as e:
        print(f"Error loading commands: {e}")
        sys.exit(1)
    
    if not commands:
        print("No commands found in the file.")
        sys.exit(1)
    
    # Setup logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True)
    
    commands_file_name = Path(args.commands).stem
    log_file = log_dir / f"{commands_file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    log_message(f"Starting command execution from: {args.commands}", log_file)
    log_message(f"Logging to: {log_file}", log_file)
    log_message(f"Total commands to run: {len(commands)}", log_file)
    
    # Show all commands
    log_message("\\nCommands to execute:", log_file)
    for i, cmd_info in enumerate(commands, 1):
        log_message(f"{i:2d}. {cmd_info['name']}: {cmd_info['command']}", log_file)
    
    # Ask for confirmation
    response = input(f"\\nRun {len(commands)} commands? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return
    
    # Run commands
    successful = 0
    failed = 0
    total_start_time = time.time()
    
    for i, cmd_info in enumerate(commands, 1):
        log_message(f"\\n{'='*60}", log_file)
        log_message(f"Executing command {i}/{len(commands)}", log_file)
        
        try:
            if run_command(cmd_info, log_file, args.continue_on_error):
                successful += 1
            else:
                failed += 1
        except (subprocess.CalledProcessError, Exception):
            failed += 1
            if not args.continue_on_error:
                log_message("Stopping execution due to error", log_file)
                break
        except KeyboardInterrupt:
            log_message("\\nExecution interrupted by user", log_file)
            break
    
    # Summary
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    log_message(f"\\n{'='*60}", log_file)
    log_message(f"Command execution completed in {total_duration:.2f}s", log_file)
    log_message(f"Successful: {successful}", log_file)
    log_message(f"Failed: {failed}", log_file)
    log_message(f"Total: {successful + failed}", log_file)
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()