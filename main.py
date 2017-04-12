import TotalActivationTool as TA

if __name__ == '__main__':
    ta = TA.TotalActivationTool()
    config = TA.Config()
    ta.load('data/data.mat', 'data/atlas.mat', config)
    ta.detrend()
    ta.regularization()
