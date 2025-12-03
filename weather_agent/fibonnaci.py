def fibonacci(n):
    a, b = 0, 1
    fib_sequence = []
    while a <= n:
        fib_sequence.append(a)
        a, b = b, a + b
    return fib_sequence

if __name__ == "__main__":
    num = int(input("Enter a number to generate Fibonacci sequence up to: "))
    result = fibonacci(num)
    print(f"Fibonacci sequence up to {num}: {result}")
