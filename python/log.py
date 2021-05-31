from termcolor import colored, cprint

class Log:
    """A class for logging.
    
    Parameters:
    level (string): one of ['DEBUG', 'INFO', 'WARN', 'ERR']
    """
    def __init__(self, level):
        if not level:
            level = "INFO"
        if level not in ["DEBUG", "INFO", "WARN", "ERR"]:
            raise ValueError(f"Unsupported logging level: {level}")

        self.level = level

    def check_level(self, level):
        """Returns true a level is high enough, given the current logging level"""
        if self.level in ["DEBUG"]:
            return True
        if level == "INFO" and self.level not in ["WARN", "ERR"]:
            return True
        if level == "WARN" and self.level not in ["ERR"]:
            return True
        if level == "ERR":
            return True


    def INFO(self, message):
        if self.check_level("INFO"):
            info_text = colored("[INFO]", "grey", attrs=["bold"])
            print(info_text, message)

    def WARN(self, message):
        if self.check_level("WARN"):
            info_text = colored("[WARN]", "yellow", attrs=["bold"])
            print(info_text, message)

    def ERR(self, message):
        if self.check_level("ERR"):
            info_text = colored("[ERR]", "red", attrs=["bold"])
            print(info_text, message)

    def DEBUG(self, message):
        if self.check_level("DEBUG"):
            info_text = colored("[DEBUG]", "blue", attrs=["bold"])
            print(info_text, message)

    

