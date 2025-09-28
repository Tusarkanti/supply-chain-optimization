import asyncio
import json
import websockets
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class IoTIntegration:
    def __init__(self):
        self.connected_sensors = {}
        self.websocket_clients = set()

    async def handle_sensor_connection(self, websocket, path):
        """Handle WebSocket connections from IoT sensors"""
        try:
            async for message in websocket:
                data = json.loads(message)
                sensor_id = data.get('sensor_id')

                if sensor_id:
                    self.connected_sensors[sensor_id] = data

                    # Broadcast to all connected clients
                    await self.broadcast_to_clients({
                        'type': 'sensor_update',
                        'sensor_id': sensor_id,
                        'data': data
                    })

                    logger.info(f"Received data from sensor {sensor_id}")
        except Exception as e:
            logger.error(f"Error handling sensor connection: {e}")

    async def broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if self.websocket_clients:
            message_json = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_json) for client in self.websocket_clients]
            )

    async def start_server(self, host: str = 'localhost', port: int = 8765):
        """Start the IoT WebSocket server"""
        async with websockets.serve(self.handle_sensor_connection, host, port):
            logger.info(f"IoT WebSocket server started on {host}:{port}")
            await asyncio.Future()  # Run forever

# Global IoT integration instance
iot_integration = IoTIntegration()
