# Predicting Sales at Favorita Stores

## Dataset Description

This project involves predicting sales for thousands of product families sold at Favorita stores in Ecuador. The dataset includes historical sales data along with various factors such as store and product information, promotional activities, and external influences like oil prices and holidays/events. The goal is to build predictive models using this rich set of features to forecast future sales accurately.

## File Descriptions and Data Field Information

### `train.csv`

- **store_nbr**: Identifier for the store selling the products.
- **family**: Type of product sold.
- **sales**: Total sales for a product family at a particular store on a given date. Fractional values are allowed.
- **onpromotion**: Number of items in a product family that were being promoted at a store on a given date.

### `test.csv`

Contains the same features as `train.csv` but requires predictions for the target sales for the 15 days following the last date in the training data.

### `sample_submission.csv`

Provides the correct format for submitting predictions.

### `stores.csv`

Includes metadata about each store:

- **city**
- **state**
- **type**
- **cluster** (grouping of similar stores)

### `oil.csv`

Daily oil prices covering both the training and testing periods. Important due to Ecuador's dependency on oil.

### `holidays_events.csv`

Details about holidays and events, including:

- **transferred** holidays (moved to another date)
- **Bridge** days (extra days added to a holiday)
- **Work Day** (days not normally scheduled for work, compensating for Bridge days)
- **Additional holidays** (added to regular calendar holidays)

## Additional Notes

- Public sector wages are paid bi-weekly on the 15th and last day of the month, potentially affecting supermarket sales.
- A significant earthquake occurred on April 16, 2016, impacting sales through relief efforts and donations.

## Getting Started

To get started with this project, clone the repository and explore the dataset. Consider the impact of promotions, oil prices, and holidays/events on sales. Use statistical analysis and machine learning techniques to develop your prediction model.

## Contributing

Contributions are welcome Please feel free to submit pull requests with improvements or new insights.
