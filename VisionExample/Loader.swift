//
//  Loader.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 11/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//
import UIKit
/*
import Foundation
// MARK: - _______________  LOADER   _______________
import SVProgressHUD

/// Show loader
public func showLoader() {
    OperationQueue.main.addOperation {
        SVProgressHUD.setDefaultMaskType(.gradient)
        SVProgressHUD.setBorderColor(UIColor.AppColor.lightingPurpleColor)
        SVProgressHUD.setForegroundColor(UIColor.AppColor.lightingPurpleColor)
        
        SVProgressHUD.show()
    }
}

/// Hide loader
public func hideLoader() {
    OperationQueue.main.addOperation {
        SVProgressHUD.dismiss()
    }
}*/
class SpinnerViewController: UIViewController {
    var spinner = UIActivityIndicatorView(style: .whiteLarge)

    override func loadView() {
        view = UIView()
        view.backgroundColor = UIColor(white: 0, alpha: 0.7)

        spinner.translatesAutoresizingMaskIntoConstraints = false
        spinner.startAnimating()
        view.addSubview(spinner)

        spinner.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        spinner.centerYAnchor.constraint(equalTo: view.centerYAnchor).isActive = true
    }
}
