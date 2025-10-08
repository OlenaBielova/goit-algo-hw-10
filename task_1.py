from typing import Dict, List
from time import perf_counter

COINS_DEFAULT = [50, 25, 10, 5, 2, 1]  

def find_coins_greedy(amount: int, coins: List[int] = None) -> Dict[int, int]:
    """
    Жадібний алгоритм для видачі решти:
    бере найбільшу доступну монету, поки можна, потім переходить до наступної.
    Повертає словник {номінал: кількість}.
    """
    if coins is None:
        coins = COINS_DEFAULT
    coins = sorted(coins, reverse=True)

    result: Dict[int, int] = {}
    remaining = amount

    for c in coins:
        if remaining <= 0:
            break
        k = remaining // c
        if k > 0:
            result[c] = k
            remaining -= k * c

    # якщо сума не збирається
    if remaining != 0:
        raise ValueError(f"Cannot make change for {amount} with given coins {coins}")

    return result

def find_min_coins(amount: int, coins: List[int] = None) -> Dict[int, int]:
    """
    Динамічне програмування:
    dp[s] - мінімальна кількість монет, щоб набрати суму s.
    prev_coin[s] - монета, якою завершуємо оптимальну комбінацію для s (для відновлення відповіді).
    Повертає словник {номінал: кількість}.
    """
    if coins is None:
        coins = COINS_DEFAULT
    coins = sorted(coins)  

    # ініціалізація
    INF = 10**9
    dp = [INF] * (amount + 1)
    prev_coin = [-1] * (amount + 1)
    dp[0] = 0

    # основний цикл: O(len(coins) * amount)
    for c in coins:
        for s in range(c, amount + 1):
            if dp[s - c] + 1 < dp[s]:
                dp[s] = dp[s - c] + 1
                prev_coin[s] = c

    if dp[amount] == INF:
        raise ValueError(f"Cannot make change for {amount} with given coins {coins}")

    # відновлюємо відповідь
    result: Dict[int, int] = {}
    s = amount
    while s > 0:
        c = prev_coin[s]
        result[c] = result.get(c, 0) + 1
        s -= c

    # для зручності відсортуємо у спадаючому
    return dict(sorted(result.items(), reverse=True))


# --- приклад використання ---

if __name__ == "__main__":
    amount = 156
    print("Сума:", amount)

    g = find_coins_greedy(amount)
    d = find_min_coins(amount)

    print("Greedy:", g)
    print("DP    :", d)         

    # Порівняння часу (оцінка, не строга)
    big_amount = 10_000
    t0 = perf_counter()
    find_coins_greedy(big_amount)
    t1 = perf_counter()
    find_min_coins(big_amount)
    t2 = perf_counter()

    print(f"Greedy time for {big_amount}: {(t1 - t0)*1000:.2f} ms")
    print(f"DP time     for {big_amount}: {(t2 - t1)*1000:.2f} ms")
