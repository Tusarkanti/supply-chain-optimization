import { io } from 'socket.io-client';

// Use environment variable for API URL, fallback to localhost for development
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

class WebSocketService {
  constructor() {
    this.socket = null;
    this.listeners = {};      // Event listeners by type
    this.eventListeners = {}; // Connect/disconnect/error
    this.connected = false;
  }

  /**
   * Connect to Socket.IO server
   * @param {string} token - JWT token for authentication
   */
  connect(token = null) {
    if (this.socket && this.connected) {
      console.log('[Socket.IO] Already connected');
      return;
    }

    this.socket = io(API_BASE_URL, {
      auth: token ? { token } : {},
      transports: ['websocket'],
      autoConnect: false,
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
    });

    // Connection events
    this.socket.on('connect', () => {
      this.connected = true;
      console.log('[Socket.IO] Connected with ID:', this.socket.id);
      this.eventListeners['connect']?.forEach(cb => cb());
    });

    this.socket.on('disconnect', (reason) => {
      this.connected = false;
      console.warn('[Socket.IO] Disconnected:', reason);
      this.eventListeners['disconnect']?.forEach(cb => cb(reason));
    });

    this.socket.on('connect_error', (err) => {
      this.connected = false;
      console.error('[Socket.IO] Connection error:', err.message);
      this.eventListeners['connect_error']?.forEach(cb => cb(err));
    });

    // Generic event listener
    this.socket.onAny((event, data) => {
      if (this.listeners[event]) {
        this.listeners[event].forEach(cb => cb(data));
      }
    });

    this.socket.connect();
  }

  /**
   * Disconnect socket
   */
  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.connected = false;
      this.socket = null;
      console.log('[Socket.IO] Disconnected manually');
    }
  }

  /**
   * Subscribe to a custom event
   * @param {string} event - Event name
   * @param {function} callback - Callback function
   */
  subscribe(event, callback) {
    if (!this.listeners[event]) this.listeners[event] = [];
    if (!this.listeners[event].includes(callback)) {
      this.listeners[event].push(callback);
      console.log(`[Socket.IO] Subscribed to event: ${event}`);
    }
  }

  /**
   * Unsubscribe from a custom event
   * @param {string} event - Event name
   * @param {function|null} callback - Specific callback to remove or null to remove all
   */
  unsubscribe(event, callback = null) {
    if (!this.listeners[event]) return;
    if (callback) {
      this.listeners[event] = this.listeners[event].filter(cb => cb !== callback);
      console.log(`[Socket.IO] Unsubscribed a callback from event: ${event}`);
    } else {
      this.listeners[event] = [];
      console.log(`[Socket.IO] Unsubscribed all callbacks from event: ${event}`);
    }
  }

  /**
   * Add event listener for connection/disconnection/error
   */
  on(event, callback) {
    if (!this.eventListeners[event]) this.eventListeners[event] = [];
    if (!this.eventListeners[event].includes(callback)) {
      this.eventListeners[event].push(callback);
      console.log(`[Socket.IO] Added listener for event: ${event}`);
    }
  }

  /**
   * Remove event listener
   */
  off(event, callback = null) {
    if (!this.eventListeners[event]) return;
    if (callback) {
      this.eventListeners[event] = this.eventListeners[event].filter(cb => cb !== callback);
      console.log(`[Socket.IO] Removed listener for event: ${event}`);
    } else {
      this.eventListeners[event] = [];
      console.log(`[Socket.IO] Removed all listeners for event: ${event}`);
    }
  }

  /**
   * Send message to server
   */
  send(event, payload) {
    if (this.socket && this.connected) {
      this.socket.emit(event, payload);
      console.log(`[Socket.IO] Sent event: ${event}`, payload);
    } else {
      console.warn('[Socket.IO] Cannot send, socket not connected');
    }
  }
}

export default new WebSocketService();
