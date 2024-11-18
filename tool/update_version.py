version = input("Update version to: ")

with open("pyproject.toml") as f:
    lines = [
        (f'version = "{version}"' if "version" in line else line).strip()
        for line in f.readlines()
    ]

with open("pyproject.toml", "w") as f:
    f.write("\n".join(lines))

with open("rmse_test/__main__.py", "w") as f:
    quotes = '"""'
    f.write(f"""
if __name__ == "__main__":
    print({quotes}
RMSE Test {version}
By Richard Nathan Ambler, November 2024
See: https://github.com/ram6ler/RMSE-Normality-Test                
{quotes})
""")
