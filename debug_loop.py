import sys
import threading
from src.brain import orchestrator

print("Starting debug orchestrator...")
orchestrator._running = True
orchestrator._main_loop()
print("Ended.")
