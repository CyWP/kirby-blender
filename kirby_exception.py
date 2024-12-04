import bpy

class KirbyError(Exception):
    """
    Automatically reports the exception to Blender's UI when raised.
    """

    def __init__(self, message, severity='ERROR'):

        super().__init__(message)
        self.message = message
        self.severity = severity.upper()  # Ensure the severity is uppercase for Blender's `report` method.

        if self.severity not in {'INFO', 'WARNING', 'ERROR'}:
            raise ValueError("Severity must be one of ['INFO', 'WARNING', 'ERROR']")

        # Automatically report the error when raised
        self.report_to_ui()

    def report_to_ui(self):

        context = bpy.context

        if context and hasattr(context, 'operator') and context.operator:
            context.operator.report({self.severity}, self.message)
        else:
            # Fallback to printing the message if no operator context is available
            print(f"{self.severity}: {self.message}")
