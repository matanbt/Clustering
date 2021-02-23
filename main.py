import user_input


def main():
    args = user_input.get_args()
    if user_input.check_user_input(args):
        params, points, clusters = user_input.generate_points(args)


if __name__ == '__main__':
    main()
