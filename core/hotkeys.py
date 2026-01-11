from pynput import keyboard


class GlobalHotkey:
    def __init__(self, on_toggle):
        self.on_toggle = on_toggle
        self.listener = None

    def start(self):
        def for_press(key):
            if key == keyboard.Key.f9:
                self.on_toggle()

        self.listener = keyboard.Listener(on_press=for_press)
        self.listener.daemon = True
        self.listener.start()

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
