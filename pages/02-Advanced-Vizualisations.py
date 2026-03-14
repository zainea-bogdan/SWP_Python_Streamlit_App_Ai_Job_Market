"""
This page is intended to show the following:
    - * Choropleth Graph - Geo discrepancies : vizs the number of roles per country/city + filters of other aspects.
        - for lat and long data below are some methods:
            - | Service                 | API key | Free tier    | Notes               |
              | ----------------------- | ------- | ------------ | ------------------- |
              | OpenStreetMap Nominatim | ❌ No    | Yes          | Very common, simple |
              | OpenCage                | ✅ Yes   | 2,500/day    | Easy JSON response  |
              | Mapbox Geocoding        | ✅ Yes   | 100k/month   | High quality        |
              | Google Geocoding        | ✅ Yes   | Limited free | Requires billing    |

    - Sunburst graphic: at industry level, to see repartition (%) of job category + number per category.
        - filter to choose the second category: job category % / job title.    
    - Sunburst graphic to vizualise % of unique skills required per job title. 
    - Interpretation Sections with general conclusions.
"""