from XInput import *


class ControllerValues:
    ### класс для транспортировки знчаений типа DTO
    def __init__(self, l_stick, r_stick, a_button, b_button, x_button, y_button,
                 dpad_right, dpad_left, dpad_up, dpad_down, right_button,
                 left_button, r_trigger, l_trigger, line_mode, qr_mode):
        self.l_stick = l_stick
        self.r_stick = r_stick
        self.a_button = a_button
        self.b_button = b_button
        self.x_button = x_button
        self.y_button = y_button
        self.dpad_right = dpad_right
        self.dpad_left = dpad_left
        self.dpad_up = dpad_up
        self.dpad_down = dpad_down
        self.right_button = right_button
        self.left_button = left_button
        self.r_trigger = r_trigger
        self.l_trigger = l_trigger
        self.line_mode = line_mode
        self.qr_mode = qr_mode


class XboxController:
    def __init__(self):
        self.l_stick = (127, 127)
        self.r_stick = (127, 127)
        self.a_button = 0
        self.b_button = 0
        self.x_button = 0
        self.y_button = 0
        self.dpad_right = 0
        self.dpad_left = 0
        self.dpad_up = 0
        self.dpad_down = 0
        self.right_button = 0
        self.left_button = 0
        self.r_trigger = 0
        self.l_trigger = 0
        self.line_mode = 0
        self.qr_mode = 0

    def read_controller(self):
        events = get_events()
        for event in events:
            if event.type == EVENT_CONNECTED:
                pass
            elif event.type == EVENT_DISCONNECTED:
                pass

            elif event.type == EVENT_STICK_MOVED:
                if event.stick == LEFT:
                    self.l_stick = (int(round(127 + (127 * event.x), 0)), int(round(127 + (127 * event.y), 0)))
                elif event.stick == RIGHT:
                    self.r_stick = (int(round(127 + (127 * event.x), 0)), int(round(127 + (127 * event.y), 0)))

            elif event.type == EVENT_BUTTON_PRESSED:
                if event.button == "A":
                    if self.a_button == 1:
                        self.a_button = 0
                    else:
                        self.a_button = 1
                elif event.button == "Y":
                    self.y_button = 1
                elif event.button == "X":
                    self.x_button = 1
                elif event.button == "B":
                    self.b_button = 1

                elif event.button == "DPAD_LEFT":
                    self.dpad_left = 1
                elif event.button == "DPAD_RIGHT":
                    self.dpad_right = 1
                elif event.button == "DPAD_UP":
                    self.dpad_up = 1
                elif event.button == "DPAD_DOWN":
                    self.dpad_down = 1
                elif event.button == "BACK":
                    self.qr_mode = (not self.qr_mode)
                elif event.button == "START":
                    self.line_mode = (not self.line_mode)


                elif event.button == "LEFT_SHOULDER":
                    self.left_button = 1
                elif event.button == "RIGHT_SHOULDER":
                    self.right_button = 1

            elif event.type == EVENT_BUTTON_RELEASED:
                if event.button == "DPAD_LEFT":
                    self.dpad_left = 0
                elif event.button == "DPAD_RIGHT":
                    self.dpad_right = 0
                elif event.button == "DPAD_UP":
                    self.dpad_up = 0
                elif event.button == "DPAD_DOWN":
                    self.dpad_down = 0
                elif event.button == "LEFT_SHOULDER":
                    self.left_button = 0
                elif event.button == "RIGHT_SHOULDER":
                    self.right_button = 0
                elif event.button == "X":
                    self.x_button = 0
                elif event.button == "B":
                    self.b_button = 0
                elif event.button == "Y":
                    self.y_button = 0

            elif event.type == EVENT_TRIGGER_MOVED:
                if event.trigger == LEFT:
                    self.l_trigger = (int(round(14 * event.value, 0)))
                if event.trigger == RIGHT:
                    self.r_trigger = (int(round(14 * event.value, 0)))

    def get_values(self):
        return ControllerValues(
            self.l_stick,
            self.r_stick,
            self.a_button,
            self.b_button,
            self.x_button,
            self.y_button,
            self.dpad_right,
            self.dpad_left,
            self.dpad_up,
            self.dpad_down,
            self.right_button,
            self.left_button,
            self.r_trigger,
            self.l_trigger,
            self.line_mode,
            self.qr_mode
        )
