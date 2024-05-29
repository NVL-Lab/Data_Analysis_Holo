import os

def get_user_choice(prompt):
    while True:
        choice = input(prompt)
        if choice in ('1', '2'):
            return choice
        else:
            print("Invalid choice. Please enter 1 or 2.")

def generate_file_path(base_dir):
    choice = get_user_choice("Choose the folder type to process (1 for raw, 2 for process): ")
    choice1 = get_user_choice("Choose the folder type to process (1 for m28, 2 for m29): ")
    choice2 = get_user_choice("Choose the folder type to process (1 for 230416, 2 for 230417): ")
    choice3 = get_user_choice("Choose the folder type to process (1 for D08, 2 for D09): ")

    folder_types = ['raw'] if choice == '1' else ['process']
    folder_types.append('m28' if choice1 == '1' else 'm29')
    folder_types.append('230416' if choice2 == '1' else '230417')
    folder_types.append('D08' if choice3 == '1' else 'D09')

    file_path = os.path.join(base_dir, *folder_types)
    return file_path

base_dir = r"C:\HoloBMI\example_data\data"
file_path = generate_file_path(base_dir)
print("Generated File Path:", file_path)
