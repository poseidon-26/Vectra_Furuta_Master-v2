"""
urdf_parser.py — Extracts physical parameters from a SolidWorks-exported URDF/XML.

Returns inertia as (ixx, iyy, izz).  Callers MUST pick the correct component
for their rotation axis:
  Joint_yaw  rotates about local Z  ("0 0 -1") → use izz = inertia[2]
  joint_pitch rotates about local Y  ("0 1  0") → use iyy = inertia[1]
"""

import math
import os
import xml.etree.ElementTree as ET


class URDFParser:

    def __init__(self, directory: str = '.'):
        valid_files = [
            f for f in os.listdir(directory)
            if f.endswith(('.urdf', '.xml'))
        ]
        if not valid_files:
            raise FileNotFoundError(
                f"FATAL: No .urdf or .xml file found in '{directory}'."
            )

        # Prefer .urdf over .xml when both are present
        urdf_path = next(
            (os.path.join(directory, f) for f in valid_files if f.endswith('.urdf')),
            os.path.join(directory, valid_files[0]),
        )
        print(f"[URDFParser] Ingesting morphology from: {urdf_path}")

        self.root = ET.parse(urdf_path).getroot()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_link_properties(self, link_name: str):
        """
        Returns (mass, com_distance, (ixx, iyy, izz)) for the named link.

        com_distance is the Euclidean norm of the inertial origin xyz — i.e.
        the distance from the joint origin to the centre of mass.

        Raises no exception on a missing link; returns safe defaults and logs a warning.
        """
        for link in self.root.findall('link'):
            if link.get('name') != link_name:
                continue

            inertial = link.find('inertial')
            if inertial is None:
                break

            mass = float(inertial.find('mass').get('value'))

            xyz = inertial.find('origin').get('xyz', '0 0 0')
            x, y, z = map(float, xyz.split())
            com_dist = math.sqrt(x**2 + y**2 + z**2)

            itag = inertial.find('inertia')
            ixx  = float(itag.get('ixx'))
            iyy  = float(itag.get('iyy'))
            izz  = float(itag.get('izz'))
            return mass, com_dist, (ixx, iyy, izz)

        print(
            f"[URDFParser] WARNING: Link '{link_name}' not found — using safe defaults."
        )
        return 0.164, 0.137, (5.17e-4, 2.595e-4, 2.596e-4)

    def get_joint_radius(self, joint_name: str) -> float:
        """
        Returns the radial distance (sqrt(x²+y²)) from the parent joint origin
        to the child joint origin — i.e. the arm length l_r.
        """
        for joint in self.root.findall('joint'):
            if joint.get('name') != joint_name:
                continue
            xyz  = joint.find('origin').get('xyz', '0 0 0')
            x, y = map(float, xyz.split()[:2])
            return math.sqrt(x**2 + y**2)

        print(
            f"[URDFParser] WARNING: Joint '{joint_name}' not found — using default l_r = 0.04 m."
        )
        return 0.040