import click


@click.command()
@click.option("-n", "--name", default="you", type=str, help="Your name")
def main(name: str) -> None:
    print(f"Hello {name}, from canlock!")


if __name__ == "__main__":
    main()
