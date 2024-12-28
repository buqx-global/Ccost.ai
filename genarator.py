import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_upc():
    """Generate valid 12-digit UPC code"""
    return f"{random.randint(100000000000, 999999999999)}"


def generate_gtin(upc):
    """Generate 13-digit GTIN from UPC by adding leading zero"""
    return f"0{upc}"


def generate_sku(product, brand, unit):
    """Generate SKU based on product characteristics"""
    # Take first 3 letters of product, brand and add random 4 digits
    prod_code = ''.join(c for c in product[:3] if c.isalnum()).upper()
    brand_code = ''.join(c for c in brand[:3] if c.isalnum()).upper()
    random_digits = str(random.randint(1000, 9999))
    return f"{prod_code}{brand_code}{random_digits}"


def generate_customer_id():
    """Generate a unique 6-digit customer ID"""
    return f"CUST{random.randint(100000, 999999)}"


def calculate_prices_and_margins(base_price):
    """
    Calculate unit cost and related financial metrics
    Ensures margin is between 0-15% using the formula:
    margin = (unit_price - unit_cost)/unit_cost * 100
    """
    target_margin = random.uniform(0, 15)  # 0-15% margin
    unit_price = round(base_price, 2)
    unit_cost = round(unit_price / (1 + target_margin / 100), 2)
    actual_margin = round((unit_price - unit_cost) / unit_cost * 100, 2)
    return unit_price, unit_cost, actual_margin


def generate_realistic_rating():
    """
    Generate more realistic product ratings using a beta distribution
    Beta distribution tends to produce right-skewed ratings common in real-world reviews

    Returns:
        float: Rating between 1-5
    """
    # Parameters for beta distribution (a=4, b=1 creates right-skewed distribution)
    rating = np.random.beta(4, 1)
    # Scale to 1-5 range and round to nearest 0.1
    return round(1 + rating * 4, 1)


def calculate_store_capacity_utilization(current_stock, max_capacity):
    """
    Calculate store capacity utilization for a product

    Args:
        current_stock: Current quantity of product in stock
        max_capacity: Maximum storage capacity for the product

    Returns:
        float: Utilization percentage (0-1)
    """
    if max_capacity <= 0:
        return 0
    return min(1.0, current_stock / max_capacity)


def verify_calculations(unit_price, unit_cost, quantity, cost_of_goods, total_price, profit, margin):
    """Verify that all financial calculations are consistent"""
    assert round(cost_of_goods, 2) == round(quantity * unit_cost, 2), "Cost of goods calculation error"
    assert round(total_price, 2) == round(quantity * unit_price, 2), "Total price calculation error"
    assert round(profit, 2) == round(total_price - cost_of_goods, 2), "Profit calculation error"
    calculated_margin = round((unit_price - unit_cost) / unit_cost * 100, 2)
    assert abs(margin - calculated_margin) < 0.01, f"Margin calculation error: {margin} vs {calculated_margin}"
    profit_based_margin = round((profit / cost_of_goods * 100), 2)
    assert abs(margin - profit_based_margin) < 0.01, f"Margin-profit inconsistency: {margin} vs {profit_based_margin}"


def get_product_info():
    """Define product categories, weights, and turnover ranges"""
    return {
        'Beverages': {
            'General': 'Consumer Goods',
            'SubCategories': ['Hot Beverages', 'Cold Beverages', 'Functional Drinks'],
            'Brands': ['Pure Life', 'Nature\'s Best', 'Organic Valley', 'Wholesome Co.'],
            'Manufacturers': ['Natural Drinks Inc', 'Global Beverage Corp', 'Organic Ventures LLC'],
            'turnover_range': (8, 12),
            'Products': {
                'Cold Pressed Juice': {
                    'weights': {
                        'lb': 0.035, 'oz': 0.54, 'g': 16.0, 'fl_oz': 0.54,
                        'ml': 16.0, 'L': 0.016, 'gal': 0.004
                    },
                    'unit': '16 oz'
                },
                'Sparkling Water 12-Pack': {
                    'weights': {
                        'lb': 0.317, 'oz': 4.87, 'g': 144.0, 'fl_oz': 4.87,
                        'ml': 144.0, 'L': 0.144, 'gal': 0.038
                    },
                    'unit': '12x12 oz'
                },
                'Premium Coffee': {
                    'weights': {
                        'lb': 0.026, 'oz': 0.41, 'g': 12.0, 'fl_oz': 0.41,
                        'ml': 12.0, 'L': 0.012, 'gal': 0.003
                    },
                    'unit': '12 oz'
                },
                'Organic Tea': {
                    'weights': {
                        'lb': 0.044, 'oz': 0.71, 'g': 20.0, 'fl_oz': None,
                        'ml': 20.0, 'L': 0.020, 'gal': 0.005
                    },
                    'unit': '20 bags'
                }
            }
        },
        'Health & Beauty': {
            'General': 'Personal Care',
            'SubCategories': ['Skincare', 'Natural Remedies', 'Oral Care'],
            'Brands': ['Pure Essence', 'Natural Care', 'Eco Beauty', 'Wellness Plus'],
            'Manufacturers': ['Natural Beauty Corp', 'Health Products Inc', 'Wellness Manufacturers'],
            'turnover_range': (4, 6),  # Lower turnover for beauty products
            'Products': {
                'Natural Shampoo': {
                    'weights': {
                        'lb': 0.035, 'oz': 0.54, 'g': 16.0, 'fl_oz': 0.54,
                        'ml': 16.0, 'L': 0.016, 'gal': 0.004
                    },
                    'unit': '16 oz'
                },
                'Organic Face Cream': {
                    'weights': {
                        'lb': 0.004, 'oz': 0.07, 'g': 2.0, 'fl_oz': 0.07,
                        'ml': 2.0, 'L': 0.002, 'gal': 0.001
                    },
                    'unit': '2 oz'
                },
                'Essential Oil Set': {
                    'weights': {
                        'lb': 0.002, 'oz': 0.04, 'g': 1.0, 'fl_oz': None,
                        'ml': 1.0, 'L': 0.001, 'gal': 0.0
                    },
                    'unit': '3x10 ml'
                }
            }
        },
        'Home & Kitchen': {
            'General': 'Household Goods',
            'SubCategories': ['Storage Solutions', 'Kitchen Tools', 'Eco-Friendly Products'],
            'Brands': ['EcoHome', 'Kitchen Pro', 'Green Living', 'Home Essentials'],
            'Manufacturers': ['HomeGoods Corp', 'Kitchen Essentials Ltd', 'Eco Solutions Inc'],
            'turnover_range': (3, 5),  # Lower turnover for durable goods
            'Products': {
                'Glass Food Containers': {
                    'weights': {
                        'lb': 0.002, 'oz': 0.04, 'g': 1.0, 'fl_oz': None,
                        'ml': 1.0, 'L': 0.001, 'gal': 0.0
                    },
                    'unit': '3 pc'
                },
                'Bamboo Utensil Set': {
                    'weights': {
                        'lb': 0.002, 'oz': 0.04, 'g': 1.0, 'fl_oz': None,
                        'ml': 1.0, 'L': 0.001, 'gal': 0.0
                    },
                    'unit': '4 pc'
                }
            }
        },
        'Grocery': {
            'General': 'Food & Beverage',
            'SubCategories': ['Organic Foods', 'Snacks', 'Dairy Alternatives'],
            'Brands': ['Nature\'s Choice', 'Organic Feast', 'Pure Harvest', 'Health Valley'],
            'Manufacturers': ['Organic Foods Corp', 'Natural Foods Ltd', 'Global Foods Inc'],
            'turnover_range': (12, 20),  # Highest turnover for groceries
            'Products': {
                'Greek Yogurt': {
                    'weights': {
                        'lb': 0.071, 'oz': 1.08, 'g': 32.0, 'fl_oz': 1.08,
                        'ml': 32.0, 'L': 0.032, 'gal': 0.008
                    },
                    'unit': '32 oz'
                },
                'Trail Mix': {
                    'weights': {
                        'lb': 0.053, 'oz': 0.81, 'g': 24.0, 'fl_oz': 0.81,
                        'ml': 24.0, 'L': 0.024, 'gal': 0.006
                    },
                    'unit': '24 oz'
                }
            }
        }
    }


def calculate_inventory_metrics(cogs_period, beginning_inventory_units, ending_inventory_units, cost_per_unit,
                                period_days=7):
    """
    Calculate inventory turnover ratio (ITR) and average inventory period (AIP)

    Args:
        cogs_period: Cost of goods sold during the period (in units sold)
        beginning_inventory_units: Inventory units at start of period
        ending_inventory_units: Inventory units at end of period
        cost_per_unit: Cost per unit of inventory
        period_days: Number of days in the period (default: 7 for weekly)

    Returns:
        tuple: (weekly_itr, annual_itr, aip_days)
    """
    # Calculate average inventory in units
    average_inventory_units = (beginning_inventory_units + ending_inventory_units) / 2

    # Calculate average inventory value
    average_inventory_value = average_inventory_units * cost_per_unit

    # Calculate COGS for the period
    cogs_value = cogs_period * cost_per_unit

    # Calculate weekly ITR
    weekly_itr = cogs_value / average_inventory_value if average_inventory_value > 0 else 0

    # Calculate annual ITR (scale up from weekly)
    annual_itr = weekly_itr * (365 / period_days)

    # Calculate monthly ITR (scale up from weekly)
    monthly_itr = weekly_itr * (30 / period_days)

    # Calculate AIP in days
    aip_days = 365 / annual_itr if annual_itr > 0 else 0

    return round(weekly_itr, 5), round(monthly_itr, 5), round(annual_itr, 5), round(aip_days, 5)


def calculate_product_max_capacity(product_data, store_type='standard'):
    """
    Calculate maximum storage capacity for a product based on:
    - Product physical dimensions (from weight/volume)
    - Standard shelf/storage dimensions
    - Store type and size

    Args:
        product_data: Dictionary containing product weights and unit information
        store_type: Type of store ('small', 'standard', 'large')

    Returns:
        int: Maximum capacity for this product
    """
    # Standard shelf dimensions (in inches)
    shelf_specs = {
        'small': {'depth': 24, 'width': 36, 'height': 72, 'shelves': 4},
        'standard': {'depth': 24, 'width': 48, 'height': 84, 'shelves': 5},
        'large': {'depth': 24, 'width': 72, 'height': 96, 'shelves': 6}
    }

    specs = shelf_specs.get(store_type, shelf_specs['standard'])

    # Calculate space per unit based on product type and weight
    weights = product_data['weights']
    unit_info = product_data['unit']

    # Estimate volume based on weight and typical product density
    volume_ml = weights['ml'] if weights['ml'] > 0 else (weights['g'] * 0.8)  # Approximate ml from grams

    # Calculate units per shelf based on product type and volume
    if 'pack' in unit_info.lower() or 'set' in unit_info.lower():
        # For multi-unit packages, consider package dimensions
        units_per_shelf = int((specs['width'] * specs['depth']) / (volume_ml * 0.1))
    else:
        # For single units
        units_per_shelf = int((specs['width'] * specs['depth']) / (volume_ml * 0.05))

    # Total capacity across all shelves
    total_capacity = units_per_shelf * specs['shelves']

    # Apply safety factor (80% of theoretical maximum)
    safe_capacity = int(total_capacity * 0.8)

    return max(safe_capacity, 50)  # Minimum capacity of 50 units


def calculate_product_cci(
        base_cci,
        product_rating,
        sales_trend,
        price_point,
        category_seasonality,
        competitor_prices=None
):
    """
    Calculate product-specific Consumer Confidence Index

    Args:
        base_cci: Base economic CCI (typically 0-200, with 100 as baseline)
        product_rating: Product rating (1-5 scale)
        sales_trend: Recent sales trend (-1 to 1, where 0 is stable)
        price_point: Current price relative to category average (1.0 is average)
        category_seasonality: Seasonal factor (0.8-1.2)
        competitor_prices: Optional list of competitor prices

    Returns:
        float: Product-specific CCI (0-200 scale)
    """
    # Convert product rating to 0-1 scale
    rating_factor = (product_rating - 1) / 4

    # Price competitiveness factor
    if competitor_prices:
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices)
        price_factor = avg_competitor_price / price_point
    else:
        price_factor = 1.0 / price_point  # Lower price = higher confidence

    # Calculate weighted product CCI
    weights = {
        'base_cci': 0.4,  # Economic conditions
        'rating': 0.2,  # Customer satisfaction
        'sales': 0.15,  # Sales momentum
        'price': 0.15,  # Price competitiveness
        'seasonal': 0.1  # Seasonal relevance
    }

    product_cci = (
            base_cci * weights['base_cci'] +
            (rating_factor * 200) * weights['rating'] +
            ((sales_trend + 1) * 100) * weights['sales'] +
            (price_factor * 100) * weights['price'] +
            (category_seasonality * 100) * weights['seasonal']
    )

    # Normalize to 0-200 range
    product_cci = max(0, min(200, product_cci))

    return round(product_cci, 2)


def get_base_cci(date):
    """
    Get base economic CCI for a given date
    This would typically come from economic data sources
    Here we simulate it based on typical patterns

    Args:
        date: datetime object

    Returns:
        float: Base CCI value (0-200)
    """
    # Simulate basic seasonal patterns
    month = date.month
    year_progress = month / 12

    # Base seasonal pattern (higher in spring/summer)
    seasonal_component = np.sin(2 * np.pi * year_progress) * 10

    # Base value around 100 with some random variation
    base = np.random.normal(100, 5)

    # Combine components
    cci = base + seasonal_component

    # Ensure within valid range
    return round(max(0, min(200, cci)), 2)


def generate_retail_data(num_records):
    """Generate complete retail dataset with all required fields"""
    product_info = get_product_info()
    data = []
    start_date = datetime(2021, 1, 1)  # Starting from 2021
    end_date = datetime(2023, 12, 31)  # Ending in 2023

    # Generate dates using normal distribution
    mean_date = start_date + (end_date - start_date) / 2
    std_days = (end_date - start_date).days / 4  # Using 1/4 of the range as std

    # Generate random dates following normal distribution
    random_days = np.random.normal(
        loc=(mean_date - start_date).days,
        scale=std_days,
        size=num_records
    )
    # Clip dates to ensure they're within range
    random_days = np.clip(random_days, 0, (end_date - start_date).days)
    dates = [start_date + timedelta(days=int(days)) for days in random_days]

    for i in range(num_records):
        # Select random product category and product
        category = np.random.choice(list(product_info.keys()))
        cat_data = product_info[category]
        product = np.random.choice(list(cat_data['Products'].keys()))
        product_data = cat_data['Products'][product]

        # Use the normally distributed date
        transaction_date = dates[i]

        # Generate expiry date based on product category
        if category == 'Grocery':
            # Shorter shelf life for groceries (3-12 months)
            expiry_days = random.randint(90, 365)
        elif category == 'Beverages':
            # Medium shelf life (6-18 months)
            expiry_days = random.randint(180, 540)
        else:
            # Longer shelf life for non-perishables (1-3 years)
            expiry_days = random.randint(365, 1095)

        expiry_date = transaction_date + timedelta(days=expiry_days)
        # Add is_expired flag
        is_expired = 1 if expiry_date < end_date else 0

        # Generate quantities and stock levels
        quantity = np.random.randint(1, 10)
        current_stock = np.random.randint(80, 600)
        seasonal_factor = round(np.random.uniform(0.8, 1.2), 2)

        # Generate pricing and financials
        base_price = round(random.uniform(5, 30), 2)
        unit_price, unit_cost, margin = calculate_prices_and_margins(base_price)

        # Calculate order totals
        # Cost for Goods down Margin Up
        cost_of_goods = round(quantity * unit_cost, 2)
        total_price = round(quantity * unit_price, 2)
        profit = round(total_price - cost_of_goods, 2)
        cogs_period = quantity * unit_cost

        beginning_inventory = current_stock

        # For ending inventory, we can estimate it as current_stock - quantity
        ending_inventory = max(0, current_stock - quantity)

        # Calculate turnover ratio
        weekly_itr, monthly_itr, annual_itr, average_inventory_period = calculate_inventory_metrics(cogs_period,
                                                                                                    beginning_inventory,
                                                                                                    ending_inventory,
                                                                                                    unit_cost)

        # Verify calculations
        verify_calculations(unit_price, unit_cost, quantity, cost_of_goods,
                            total_price, profit, margin)

        # Get weights from product data
        weights = product_data['weights']

        capacity_utilization = calculate_product_max_capacity(product_data)

        # Get base economic CCI for the date
        base_cci = get_base_cci(transaction_date)

        # Calculate product-specific CCI
        product_cci = calculate_product_cci(
            base_cci=base_cci,
            product_rating=generate_realistic_rating(),
            sales_trend=0,
            price_point=unit_price / cat_data.get('avg_category_price', unit_price),
            category_seasonality=seasonal_factor
        )

        upc = generate_upc()
        gtin = generate_gtin(upc)
        sku = generate_sku(product, cat_data['Brands'][0], product_data['unit'])

        record = {
            'Invoice_ID': f"US{random.randint(100000, 999999)}",
            'Customer_ID': generate_customer_id(),
            'Branch': random.choice(['Central', 'Northeast', 'Southeast', 'West']),
            'City': random.choice(['Philadelphia', 'Chicago', 'San Diego', 'Boston', 'New York',
                                   'Los Angeles', 'Houston', 'Phoenix', 'Dallas', 'San Antonio',
                                   'San Jose', 'Austin']),
            'Customer_Type': random.choice(['Regular', 'Member']),
            'Gender': random.choice(['Male', 'Female']),
            'Product': product,
            'Product_General_Category': cat_data['General'],
            'Product_Category': category,
            'Product_SubCategory': random.choice(cat_data['SubCategories']),
            'Product_Brand': random.choice(cat_data['Brands']),
            'Manufacturer': random.choice(cat_data['Manufacturers']),
            'Unit': product_data['unit'],
            'Unit_Cost': unit_cost,
            'Current_Unit_Price': unit_price,
            'Proposed_Unit_Price': None,
            'New_Unit_Price': None,
            'Quantity': quantity,
            'Cost_of_Goods': cost_of_goods,
            'current_Margin': margin,
            'new_Margin': None,
            'Profit': profit,
            'product_cci': product_cci,
            'Date': transaction_date.strftime('%Y-%m-%d'),
            'Time': f"{random.randint(8, 22):02d}:{random.randint(0, 60):02d}",
            'Payment_Method': random.choice(['Cash', 'Credit Card', 'Debit Card', 'Digital Wallet']),
            'Rating': round(random.uniform(1, 5), 1),
            'Customer_Age_Group': random.choice(['18-24', '25-34', '35-44', '45-54', '55+']),
            'Beginning Inventory Stock': beginning_inventory,
            'Ending Inventory Stock': ending_inventory,
            'Current_Stock': current_stock,
            'Reorder_Point': random.randint(16, 120),
            'Weekly_itr': weekly_itr,
            'Monthly_itr': monthly_itr,
            'Annual_itr': annual_itr,
            'Turnover_Ratio_Change_Flag': None,
            'Average_inventory_period': average_inventory_period,
            'Seasonal_Factor': seasonal_factor,
            'Seasonal_Factor_Change_Flag': None,
            'Temperature': round(random.uniform(0, 120), 1),
            'Precipitation': round(random.uniform(0, 10), 2),
            'Humidity': round(random.uniform(0, 100), 1),
            'Holiday': '',
            'Is_Holiday': random.choice([0, 1]),
            'Days_To_Next_Holiday': random.randint(0, 30),
            'Day_Of_Week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][
                transaction_date.weekday()],
            'Is_Weekend': 1 if transaction_date.weekday() >= 5 else 0,
            'Days_To_Payday': random.randint(0, 15),
            'Local_Events': '',
            'Competitor_Promotion': random.choice([True, False]),
            'Marketing_Campaign': random.choice([True, False]),
            'Consumer_Confidence_Index': round(random.uniform(0, 150), 2),
            'Store_Capacity_Utilization': capacity_utilization,
            'Traffic_Level': round(random.uniform(0, 1), 3),
            'Traffic_Level_Change_Flag': None,
            'Online_Competition_Index': round(random.uniform(0, 150), 2),
            'Universal_Product_Code': upc,
            'SKU': sku,
            'GTIN': gtin,
            'Weight_lb': weights['lb'],
            'Weight_oz': weights['oz'],
            'Weight_g': weights['g'],
            'Weight_fl_oz': weights['fl_oz'] if weights['fl_oz'] is not None else 0,
            'Weight_ml': weights['ml'],
            'Weight_L': weights['L'],
            'Weight_gallon': weights['gal'],
            'Weight_all': f"{weights['g']}g/{weights['ml']}ml",
            'Expiry_Date': expiry_date.strftime('%Y-%m-%d'),
            'Is_Expired': is_expired,
        }
        data.append(record)

    df = pd.DataFrame(data)
    # Sort by date to ensure chronological order
    df = df.sort_values(['Date', 'Product_Category'])

    # Reorder columns
    cols_to_group = ['Product_Category', 'Product', 'Product_General_Category',
                     'Product_SubCategory', 'Product_Brand', 'Manufacturer']
    cols = [col for col in df.columns if col not in cols_to_group]
    new_cols = cols[:5] + cols_to_group + cols[5:]
    df = df[new_cols]

    return df


if __name__ == "__main__":
    df = generate_retail_data(3000)
    df.to_csv('valid_retail_data1.csv', index=False)
