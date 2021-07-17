class Outer:
    def __init__(self):
        self.name = "Andy"
        self.inner = self.Inner(self)

    class Inner:
        def __init__(self, outer_parent):
            print(outer_parent.file_list)
            pass


ghost = [1, 2, 3, 4, 5]
Outer(ghost)
