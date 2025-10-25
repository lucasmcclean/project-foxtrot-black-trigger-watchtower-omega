import asyncio
import websockets
import json
import random

HOST = 'localhost'
PORT = 8765

def generate_payload():
    """Generate a random game action payload"""
    payload = {
        "move": [random.choice([-1, 0, 1]), random.choice([-1, 0, 1])],  # x, y movement
        "jump": random.choice([True, False]),
        "punch": random.choice([True, False]),
        "kick": random.choice([True, False]),
        "flash_step": random.choice([True, False])
    }
    return payload

async def handler(websocket):
    while True:
        payload = generate_payload()
        await websocket.send(json.dumps(payload))
        await asyncio.sleep(0.1)  # 10 Hz

async def main():
    async with websockets.serve(handler, HOST, PORT):
        print(f"WebSocket server running at ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())

