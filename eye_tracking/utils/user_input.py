def get_user_input(prompt):
    """
    Prompts the user with a yes/no question and returns the user's input.
    Loops until the user enters a valid input.

    Returns
    -------
    bool
        True if the user enters 1, False if the user enters 0.
    """
    prompt = prompt + ' Enter 1 (yes) or 0 (no):'
    invalid_input = 'Invalid input. Please enter "1" or "0".'

    while True:
        try:
            user_input = int(input(prompt))

        except ValueError:
            print(invalid_input)
            continue

        if user_input == 1:
            return True
        
        elif user_input == 0:
            return False
        
        else:
            print(invalid_input)