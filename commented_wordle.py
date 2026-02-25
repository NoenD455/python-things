import random

def get_feedback(guess, target):
    """
    Returns a 5-character string of R/?/# based on the guess compared to the target.
    """
    # Work with mutable lists so we can mark used digits
    target_list = list(target)
    guess_list = list(guess)
    feedback = [''] * 5

    # First pass: exact matches (R)
    for i in range(5):
        if guess_list[i] == target_list[i]:
            feedback[i] = 'R'
            target_list[i] = None          # mark this target digit as used

    # Second pass: correct digit in wrong position (?)
    for i in range(5):
        if feedback[i] == '':               # not already marked as exact
            if guess_list[i] in target_list:
                feedback[i] = '?'
                # remove the first occurrence of this digit from target_list
                idx = target_list.index(guess_list[i])
                target_list[idx] = None
            else:
                feedback[i] = '#'

    return ''.join(feedback)


def play_game():
    """One full game: generate a target and let the player guess up to 6 times."""
    target = ''.join(str(random.randint(0, 9)) for _ in range(5))
    attempts = 0
    max_attempts = 6

    print("\n--- New Game ---")
    print("Guess the 5-digit number. You have 6 tries.")

    while attempts < max_attempts:
        guess = input(f"Attempt {attempts + 1}: ").strip()

        # Validate input: exactly 5 digits
        if len(guess) != 5 or not guess.isdigit():
            print("Invalid input. Please enter exactly 5 digits (0-9).")
            continue

        attempts += 1
        feedback = get_feedback(guess, target)
        print(feedback)

        if guess == target:
            print("Congratulations! You guessed it!")
            return True

    print(f"Sorry, you ran out of tries. The number was {target}.")
    return False


def main():
    """Main loop: play games until the player chooses to quit."""
    play_again = True
    while play_again:
        play_game()
        response = input("Play again? (y/n): ").strip().lower()
        play_again = (response == 'y')
    print("Thanks for playing!")


if __name__ == "__main__":
    main()