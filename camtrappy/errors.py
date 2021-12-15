class UnresolvedFoldersError(Exception):
    message = 'Expected folder {missing_folder}. \
               Unresolved folders: {remaining_folders}'

    def __init__(self, missing_folder, remaining_folders):
        self.missing_folder = missing_folder
        self.remaining_folders = remaining_folders
        self.message = f'Expected a name for folder placeholder ' \
                       f'`{missing_folder}`. ' \
                       f'Unresolved folders: {remaining_folders}'
        super().__init__(self.message)

    def __str__(self):
        return self.message