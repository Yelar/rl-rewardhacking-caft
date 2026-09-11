#!/usr/bin/env python3

"""Compare a live AWS Price List response with a reviewed sanitized snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def extract_on_demand_price(payload: dict[str, Any]) -> dict[str, str]:
    matches: list[dict[str, str]] = []
    price_list = payload.get("PriceList")
    if not isinstance(price_list, list):
        raise ValueError("AWS Price List response has no PriceList array")
    for encoded in price_list:
        item = json.loads(encoded) if isinstance(encoded, str) else encoded
        product = item.get("product", {})
        attributes = product.get("attributes", {})
        if not (
            attributes.get("instanceType") == "p4d.24xlarge"
            and attributes.get("regionCode") == "us-east-1"
            and attributes.get("location") == "US East (N. Virginia)"
            and attributes.get("operatingSystem") == "Linux"
            and attributes.get("tenancy") == "Shared"
            and attributes.get("preInstalledSw") == "NA"
            and attributes.get("capacitystatus") == "Used"
            and attributes.get("marketoption") == "OnDemand"
            and attributes.get("operation") == "RunInstances"
        ):
            continue
        terms = item.get("terms", {}).get("OnDemand", {})
        for term in terms.values():
            for dimension in term.get("priceDimensions", {}).values():
                if dimension.get("unit") != "Hrs":
                    continue
                matches.append(
                    {
                        "capacity_status": attributes["capacitystatus"],
                        "currency": "USD",
                        "effective_date": term["effectiveDate"],
                        "instance_type": attributes["instanceType"],
                        "location": attributes["location"],
                        "market_option": attributes["marketoption"],
                        "operating_system": attributes["operatingSystem"],
                        "price_per_hour": dimension["pricePerUnit"]["USD"],
                        "pricing_api_region": "us-east-1",
                        "publication_date": item["publicationDate"],
                        "region_code": attributes["regionCode"],
                        "sku": product["sku"],
                        "tenancy": attributes["tenancy"],
                        "unit": dimension["unit"],
                    }
                )
    if len(matches) != 1:
        raise ValueError(f"Expected one exact P4d On-Demand hourly price, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--price-cap", type=float, required=True)
    args = parser.parse_args()
    live = extract_on_demand_price(json.load(sys.stdin))
    reviewed = json.loads(args.snapshot.read_text(encoding="utf-8"))
    if live != reviewed:
        raise ValueError("Live P4d Price List entry differs from the reviewed snapshot")
    price = float(live["price_per_hour"])
    if price > args.price_cap:
        raise ValueError(f"Live P4d price {price} exceeds reviewed cap {args.price_cap}")
    print(json.dumps({"price_per_hour_usd": price, "verified": True}, sort_keys=True))


if __name__ == "__main__":
    main()
