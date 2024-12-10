import numpy as np
import struct


class CameraModel:
    def __init__(self):
        self.K = None  # Intrinsic matrix
        self.D = None  # Distortion coefficients
        self.R = None  # Rotation matrix
        self.P = None  # Projection matrix
        self.local_transform = None  # Local transform
        self.image_size = (0, 0)  # Width, Height

    def deserialize_calibration(self, data):
        """
        Deserialize camera calibration data from a byte buffer.

        :param data: Byte buffer containing serialized calibration data
        :return: Number of bytes processed
        """
        # Ensure minimum data size for header
        header_size = 11
        if len(data) < header_size * 4:  # 4 bytes per int
            raise ValueError(
                f"Insufficient data size. Expected at least {header_size * 4} bytes."
            )

        # Unpack header
        header = struct.unpack("11i", data[: header_size * 4])
        print("Header:", header)

        # Extract header components
        version_major, version_minor, version_patch = header[0:3]
        data_type = header[3]
        print("Version:", version_major, version_minor, version_patch)

        # Check data type (0 for mono camera)
        if data_type != 0:
            raise ValueError(f"Unsupported calibration type: {data_type}")

        # Set image size
        self.image_size = (header[4], header[5])
        print("Image Size:", self.image_size)

        # Indices for different matrix components
        iK, iD, iR, iP, iL = 6, 7, 8, 9, 10

        # Calculate required data size
        required_data_size = (
            header_size * 4  # Header size in bytes
            + 8
            * (
                header[iK] + header[iD] + header[iR] + header[iP]
            )  # Double precision matrices
            + 4 * header[iL]  # Local transform (float)
        )

        # Verify total data size
        if len(data) < required_data_size:
            raise ValueError(
                f"Insufficient data size. "
                f"Actual: {len(data)} bytes, "
                f"Required: {required_data_size} bytes"
            )

        # Current index in the data buffer
        index = header_size * 4

        # Parse Intrinsic Matrix (K)
        if header[iK] != 0:
            assert header[iK] == 9, "K matrix must be 3x3"
            K_bytes = data[index : index + 8 * 9]
            self.K = np.frombuffer(K_bytes, dtype=np.float64).reshape((3, 3))
            index += 8 * 9
            print("Intrinsic Matrix (K):", self.K)

        # Parse Distortion Coefficients (D)
        if header[iD] != 0:
            D_bytes = data[index : index + 8 * header[iD]]
            self.D = np.frombuffer(D_bytes, dtype=np.float64).reshape((1, header[iD]))
            index += 8 * header[iD]
            print("Distortion Coefficients (D):", self.D)

        # Parse Rotation Matrix (R)
        if header[iR] != 0:
            assert header[iR] == 9, "R matrix must be 3x3"
            R_bytes = data[index : index + 8 * 9]
            self.R = np.frombuffer(R_bytes, dtype=np.float64).reshape((3, 3))
            index += 8 * 9
            print("Rotation Matrix (R):", self.R)

        # Parse Projection Matrix (P)
        if header[iP] != 0:
            assert header[iP] == 12, "P matrix must be 3x4"
            P_bytes = data[index : index + 8 * 12]
            self.P = np.frombuffer(P_bytes, dtype=np.float64).reshape((3, 4))
            index += 8 * 12
            print("Projection Matrix (P):", self.P)

        # Parse Local Transform
        if header[iL] != 0:
            assert header[iL] == 12, "Local transform must be 12 elements"
            local_transform_bytes = data[index : index + 4 * 12]
            self.local_transform = np.frombuffer(
                local_transform_bytes, dtype=np.float32
            )
            index += 4 * 12
            print("Local Transform:", self.local_transform)

        return index
