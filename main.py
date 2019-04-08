from data import attributes

while True:
    choice = str(input('\nDisplay Attribute Info? (y/n): ')).lower()
    if choice == 'q':
        break
    elif choice == 'y':
        for attribute in attributes:
            print(f'{attribute.name} - {attribute.description}')
    else:
        print('you have not entered a viable option.')
