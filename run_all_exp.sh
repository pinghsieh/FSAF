# Copyright (c) 2021
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# model that in log file
# args -is (number of initial sample), -det(using max function), -cuda(device id)
model_name="FSAF"
cuda_id=0

# optimization benchmark functions
# # args -lsH(length scale uppde bound), -lsL(length scale lower bound)
python evaluate_fsaf_blackbox.py -m ${model_name} -cuda ${cuda_id} -det True

# # # kernel function
python evaluate_fsaf_gps.py -m ${model_name} -lsH 0.55 -lsL 0.5 -det True -cuda ${cuda_id}  -is 10

# # HPO
# # args -ls(GP model length scales for each dimension)
python evaluate_fsaf_hpo.py -m ${model_name} -d "hpobenchXGB" -dl 48 -dim 6 -cuda ${cuda_id} -det 1 -is 15 -ls 11.870 0.787 6.060 10.142 11.142 10.255

python evaluate_fsaf_hpo.py -m ${model_name} -d "pm25" -dl 30 -dim 14 -cuda ${cuda_id} -det 1 -is 15 -ls 14.358 12.487 0.707 11.774 24.295 15.115 26.688 21.721 16.819 18.163 22.959 26.496 14.655 11.758

python evaluate_fsaf_hpo.py -m ${model_name} -d "augment" -dl 40 -dim 12 -cuda ${cuda_id} -det 1 -is 15 -ls 7.779 5.257 6.115 6.889 8.600 5.782 9.606 9.454 0.772 1.013 1.686 1.652

python evaluate_fsaf_hpo.py -m ${model_name} -d "Asteroid" -dl 40 -dim 12 -cuda ${cuda_id} -det 1 -is 5 -ls 34.281 79.165 48.592 134.137 110.995 3.629 41.748 12.000 9.781 30.804 50.327 100.637

python evaluate_fsaf_hpo.py -m ${model_name} -d "oil" -dl 30 -dim 4 -cuda ${cuda_id} -det 1 -is 10 -ls 0.436 0.1 0.723 0.1




