"""Preflight for this observed local RTX/Fabric camera configuration only."""
def require_supported_camera_device(argv):
    if '--enable_cameras' not in argv:
        return
    if '--device' not in argv or argv[argv.index('--device')+1] != 'cuda:0':
        raise ValueError('local captured RTX/SelectPrims path requires logical cuda:0; reject unsupported camera launch before process/physics')
