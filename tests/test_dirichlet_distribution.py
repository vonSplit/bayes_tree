import sys
import os

# Path to the built module
build_python = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'build', 'python'))
print("Adding to sys.path:", build_python)
sys.path.insert(0, build_python)

print("Imported pybayes_tree OK")
from pybayes_tree import DirichletDistribution # pyright: ignore[reportMissingImports]

def test_log_pdf_mean_higher_than_extreme():
    alpha = [2.0, 3.0, 5.0]
    d = DirichletDistribution(alpha)
    mean_point = d.mean()
    extreme_point = [0.01, 0.01, 0.98]

    logpdf_mean = d.log_pdf(mean_point)
    logpdf_extreme = d.log_pdf(extreme_point)

    print(f"logpdf(mean) = {logpdf_mean}, logpdf(extreme) = {logpdf_extreme}")
    assert logpdf_mean > logpdf_extreme
