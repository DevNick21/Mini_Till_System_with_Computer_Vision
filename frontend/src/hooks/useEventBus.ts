// Simple event bus for cross-component communication
type EventCallback = (data?: any) => void;

class EventBus {
  private events: { [key: string]: EventCallback[] } = {};

  on(event: string, callback: EventCallback) {
    if (!this.events[event]) {
      this.events[event] = [];
    }
    this.events[event].push(callback);
  }

  off(event: string, callback: EventCallback) {
    if (!this.events[event]) return;
    this.events[event] = this.events[event].filter(cb => cb !== callback);
  }

  emit(event: string, data?: any) {
    if (!this.events[event]) return;
    this.events[event].forEach(callback => callback(data));
  }
}

// Global event bus instance
export const eventBus = new EventBus();

// Custom hook for using the event bus
export const useEventBus = () => {
  return {
    emit: (event: string, data?: any) => eventBus.emit(event, data),
    on: (event: string, callback: EventCallback) => eventBus.on(event, callback),
    off: (event: string, callback: EventCallback) => eventBus.off(event, callback),
  };
};

// Event types
export const EVENTS = {
  BET_UPLOADED: 'bet_uploaded',
  ALERT_RESOLVED: 'alert_resolved',
  REFRESH_ALERTS: 'refresh_alerts',
} as const;