"""
dicom_reader.py
DICOM 시리즈 폴더 및 단일 의료영상 파일(DICOM, MHA, NIfTI 등) 로더.
"""

import os
import sys
from typing import Tuple, Dict, Any, Optional
from contextlib import contextmanager
import SimpleITK as sitk
import pydicom
import vtk

from dental_fea_app.imaging.volume_data import itk_to_vtk_image


@contextmanager
def suppress_c_stderr():
    """
    SimpleITK / GDCM / libjpeg 등의 C 라이브러리가 stderr(fd 2)로 직접 출력하는
    'Unsupported JPEG data precision 12' 등의 콘솔 노이즈 메시지를 임시로 차단합니다.
    """
    try:
        sys.stderr.flush()
        null_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
    except Exception:
        yield


class DicomReader:
    """
    치과 CBCT/CT 및 의료 영상 파일을 로드하여 정규화된 VTK ImageData로 변환하는 클래스.
    """

    @staticmethod
    def load_dicom_series(folder_path: str) -> Tuple[vtk.vtkImageData, Dict[str, Any]]:
        """
        DICOM 파일이 포함된 디렉터리를 읽어 3D VTK ImageData 및 메타데이터를 반환합니다.
        하위 디렉터리(예: DICOMDIR과 병존하는 서브폴더)까지 자동으로 탐색합니다.
        """
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(f"존재하지 않는 폴더입니다: {folder_path}")

        sitk.ProcessObject.SetGlobalWarningDisplay(False)
        reader = sitk.ImageSeriesReader()

        # 1. 대상 폴더 및 하위 폴더 재귀 탐색으로 DICOM 시리즈 목록 수집
        best_files = []
        best_dir = folder_path

        # 먼저 해당 폴더 자체 검사
        candidate_dirs = [folder_path]
        for root, dirs, _ in os.walk(folder_path):
            for d in dirs:
                candidate_dirs.append(os.path.join(root, d))

        with suppress_c_stderr():
            for c_dir in candidate_dirs:
                try:
                    s_ids = reader.GetGDCMSeriesIDs(c_dir)
                    for s_id in s_ids:
                        fnames = list(reader.GetGDCMSeriesFileNames(c_dir, s_id))
                        # 파일만 필터링
                        fnames = [f for f in fnames if os.path.isfile(f)]
                        if len(fnames) > len(best_files):
                            best_files = fnames
                            best_dir = c_dir
                except Exception:
                    continue

            # 2. GDCM이 시리즈를 찾지 못한 경우, 직접 파일 시그니처 검사 (Fallback)
            if not best_files:
                candidate_files = []
                for root, _, files in os.walk(folder_path):
                    for f in files:
                        full_p = os.path.join(root, f)
                        if not os.path.isfile(full_p):
                            continue
                        # DICOM 파일 매직 넘버 검사 (오프셋 128의 'DICM')
                        try:
                            with open(full_p, 'rb') as fp:
                                header = fp.read(132)
                                if len(header) >= 132 and header[128:132] == b'DICM':
                                    candidate_files.append(full_p)
                                elif f.lower().endswith(('.dcm', '.ima')):
                                    candidate_files.append(full_p)
                        except Exception:
                            pass

                if candidate_files:
                    # 파일명/번호 기준 정렬
                    candidate_files.sort()
                    best_files = candidate_files

            if not best_files:
                raise ValueError(f"폴더 내에서 유효한 DICOM 영상 시리즈를 찾을 수 없습니다: {folder_path}")

            # SimpleITK로 시리즈 로드
            reader.SetFileNames(best_files)
            itk_image = reader.Execute()

        # pydicom으로 대표 메타데이터 추출
        dicom_meta = {}
        try:
            if best_files:
                ds = pydicom.dcmread(best_files[0], stop_before_pixels=True)
                dicom_meta["PatientName"] = str(getattr(ds, "PatientName", "Unknown"))
                dicom_meta["PatientID"] = str(getattr(ds, "PatientID", "Unknown"))
                dicom_meta["Modality"] = str(getattr(ds, "Modality", "CT"))
                dicom_meta["SliceThickness"] = float(getattr(ds, "SliceThickness", 1.0))
                dicom_meta["PixelSpacing"] = list(map(float, getattr(ds, "PixelSpacing", [1.0, 1.0])))
        except Exception:
            pass

        vtk_image, meta = itk_to_vtk_image(itk_image)
        meta["source_type"] = "dicom_series"
        meta["folder_path"] = best_dir
        meta["file_count"] = len(best_files)
        meta.update(dicom_meta)

        return vtk_image, meta

    @staticmethod
    def load_single_file(file_path: str) -> Tuple[vtk.vtkImageData, Dict[str, Any]]:
        """
        단일 볼륨 파일 (.dcm, .mha, .mhd, .nii, .nii.gz, .nrrd 등)을 읽어 반환합니다.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"존재하지 않는 파일입니다: {file_path}")

        with suppress_c_stderr():
            itk_image = sitk.ReadImage(file_path)
        vtk_image, meta = itk_to_vtk_image(itk_image)

        meta["source_type"] = "single_file"
        meta["file_path"] = file_path

        # 확장자가 dcm인 경우 pydicom 메타데이터 시도
        if file_path.lower().endswith(('.dcm', '.ima')):
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                meta["PatientName"] = str(getattr(ds, "PatientName", "Unknown"))
                meta["Modality"] = str(getattr(ds, "Modality", "CT"))
            except Exception:
                pass

        return vtk_image, meta

    @classmethod
    def load_path(cls, path: str) -> Tuple[vtk.vtkImageData, Dict[str, Any]]:
        """
        경로가 디렉터리인지 파일인지 판별하여 적절한 로더를 호출합니다.
        """
        if os.path.isdir(path):
            return cls.load_dicom_series(path)
        elif os.path.isfile(path):
            return cls.load_single_file(path)
        else:
            raise FileNotFoundError(f"유효하지 않은 경로입니다: {path}")
