"""Tests for kinetic.constants — zone/region/location helpers."""

import os
from unittest import mock

from absl.testing import absltest, parameterized

from kinetic.constants import (
  DEFAULT_REGION,
  DEFAULT_ZONE,
  get_default_zone,
  zone_to_ar_location,
  zone_to_region,
)


class TestZoneToRegion(parameterized.TestCase):
  @parameterized.parameters(
    ("us-central1-a", "us-central1"),
    ("us-central1-b", "us-central1"),
    ("us-east1-b", "us-east1"),
    ("us-east4-c", "us-east4"),
    ("us-west1-a", "us-west1"),
    ("us-west4-b", "us-west4"),
    ("europe-west1-b", "europe-west1"),
    ("europe-west4-b", "europe-west4"),
    ("asia-east1-c", "asia-east1"),
    ("asia-southeast1-a", "asia-southeast1"),
    ("me-west1-a", "me-west1"),
    ("southamerica-east1-b", "southamerica-east1"),
  )
  def test_zone_to_region(self, zone, expected_region):
    self.assertEqual(zone_to_region(zone), expected_region)

  @parameterized.parameters(
    ("",),
    (None,),
    ("invalid",),
  )
  def test_fallback_returns_default(self, zone):
    self.assertEqual(zone_to_region(zone), DEFAULT_REGION)


class TestZoneToArLocation(parameterized.TestCase):
  @parameterized.parameters(
    ("us-central1-a", "us"),
    ("us-east1-b", "us"),
    ("us-west1-a", "us"),
    ("europe-west1-b", "europe"),
    ("europe-west4-b", "europe"),
    ("asia-east1-c", "asia"),
    ("asia-southeast1-a", "asia"),
    ("me-west1-a", "me"),
    ("southamerica-east1-b", "southamerica"),
  )
  def test_zone_to_ar_location(self, zone, expected_location):
    self.assertEqual(zone_to_ar_location(zone), expected_location)


class TestGetDefaultZone(parameterized.TestCase):
  @parameterized.parameters(
    ("us-west1-b", "us-west1-b"),
    ("europe-west4-a", "europe-west4-a"),
    ("asia-east1-c", "asia-east1-c"),
  )
  def test_returns_env_var_when_set(self, env_value, expected_zone):
    with mock.patch.dict(os.environ, {"KERAS_REMOTE_ZONE": env_value}):
      self.assertEqual(get_default_zone(), expected_zone)

  def test_returns_default_when_unset(self):
    env = {k: v for k, v in os.environ.items() if k != "KERAS_REMOTE_ZONE"}
    with mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(get_default_zone(), DEFAULT_ZONE)

  @parameterized.parameters(
    (DEFAULT_ZONE, "us-central1-a"),
    (DEFAULT_REGION, "us-central1"),
  )
  def test_default_constants(self, constant, expected_value):
    self.assertEqual(constant, expected_value)


if __name__ == "__main__":
  absltest.main()
