import nibabel as nib
import numpy as np

ORIENTATION_REF = "RAS" 


def get_orientation_from_header(header):
    aff = header.get_best_affine()
    return "".join(nib.aff2axcodes(aff))


def reorient_nifti(input_path, output_path):

    nii = nib.load(input_path)
    data = nii.get_fdata()
    header = nii.header

    src_orientation = get_orientation_from_header(header)
    print(src_orientation)
    if src_orientation!=ORIENTATION_REF:

        ornt_transf = nib.orientations.ornt_transform(
            nib.orientations.axcodes2ornt(tuple(src_orientation)),
            nib.orientations.axcodes2ornt(tuple(ORIENTATION_REF)))

        new_data = nib.orientations.apply_orientation(data, ornt_transf)

        src_affine = header.get_best_affine()
        aff_change = nib.orientations.inv_ornt_aff(ornt_transf, data.shape)
        new_affine = src_affine @ aff_change

        new_header = header.copy()
        new_header.set_qform(new_affine)
        new_header.set_sform(new_affine)
        new_header.set_data_shape(new_data.shape)

        nib.save(nib.Nifti1Image(new_data, new_affine, new_header), output_path)

        return output_path

if __name__ == "__main__":
    reorient_nifti("/home/ge.polymtl.ca/p123239/data/ms-multi-spine-challenge-2024/sub-001/anat/sub-001_T2w.nii.gz", "/home/ge.polymtl.ca/p123239/data/ms-multi-spine-challenge-2024/sub-001/anat/sub-001_T2w.nii.gz")