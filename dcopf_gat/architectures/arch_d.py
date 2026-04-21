@register("D")
class ArchD(ArchC):
    def prepare_data(self, train_x, train_y, val_x, val_y, test_x, test_y):
        w = self.cfg.window
        if w and w > 1:
            train_x, train_y = make_windows_concat(train_x, train_y, window=w)
            val_x, val_y = make_windows_concat(val_x, val_y, window=w)
            test_x, test_y = make_windows_concat(test_x, test_y, window=w)
        return (train_x, train_y), (val_x, val_y), (test_x, test_y)