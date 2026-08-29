"""Generate synthetic e-commerce order data with realistic return-risk correlations."""
import numpy as np
import pandas as pd
import random
import argparse
import os


def generate_orders(n, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    categories = ['apparel', 'electronics', 'home_goods', 'beauty', 'footwear', 'books']
    payment_methods = ['UPI', 'credit_card', 'debit_card', 'netbanking', 'cod']
    category_base_return_rate = {
        'apparel': 0.28, 'footwear': 0.25, 'beauty': 0.15,
        'electronics': 0.08, 'home_goods': 0.10, 'books': 0.03
    }

    rows = []
    for i in range(n):
        category = random.choice(categories)
        payment_method = random.choice(payment_methods)
        order_amount = round(np.random.lognormal(mean=6.5, sigma=0.8), 2)
        discount_pct = round(np.random.choice([0,5,10,15,20,30,40,50],
                                                 p=[0.25,0.15,0.15,0.15,0.1,0.1,0.06,0.04]), 0)
        customer_past_orders = np.random.poisson(3)
        customer_past_returns = min(np.random.poisson(0.5), customer_past_orders)
        customer_return_rate = customer_past_returns / customer_past_orders if customer_past_orders > 0 else 0
        account_age_days = random.randint(1, 1500)
        day_of_week = random.randint(0, 6)
        hour_of_day = random.randint(0, 23)
        address_mismatch = random.choices([0, 1], weights=[0.9, 0.1])[0]

        prob = category_base_return_rate[category]
        prob += 0.15 if discount_pct >= 40 else (0.05 if discount_pct >= 20 else 0)
        prob += 0.10 * customer_return_rate
        prob -= 0.03 * min(customer_past_orders, 5) / 5
        prob += 0.05 if address_mismatch == 1 else 0
        prob += 0.04 if payment_method == 'cod' else 0
        prob -= 0.02 if account_age_days > 365 else 0
        prob += np.random.normal(0, 0.03)
        prob = np.clip(prob, 0.01, 0.95)

        returned = np.random.binomial(1, prob)

        rows.append({
            'order_id': f"ORD{100000+i}", 'customer_id': f"CUST{random.randint(1000, n//2+1000)}",
            'category': category, 'payment_method': payment_method, 'order_amount': order_amount,
            'discount_pct': discount_pct, 'customer_past_orders': customer_past_orders,
            'customer_past_returns': customer_past_returns,
            'customer_return_rate': round(customer_return_rate, 3),
            'account_age_days': account_age_days, 'day_of_week': day_of_week,
            'hour_of_day': hour_of_day, 'address_mismatch': address_mismatch, 'returned': returned
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rows', type=int, default=18000)
    parser.add_argument('--output', type=str, default='data/raw_orders.csv')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outtput), exist_ok=True)
    df = generate_orders(args.n_rows)
    df.to_csv(args.output, index=False)

    print(f"✅ Generated {len(df)} orders -> {args.output}")
    print(f"Return rate: {df['returned'].mean():.2%}")
